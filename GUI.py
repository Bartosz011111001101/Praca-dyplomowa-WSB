import os
import numpy as np
import joblib
import tensorflow as tf
import soundfile as sf
from scipy.signal import get_window
from scipy.fft import rfft
from scipy.fftpack import dct
import librosa
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, QObject 
# Używam QObject i QThread, aby umożliwić emitowanie sygnałów w wątku roboczym
from Panel.panel_ui import Ui_MainWindow  # import wygenerowanego UI
import serial
import wave
import pandas as pd
from openai import OpenAI
import base64
from tkinter import Tk, filedialog


TEST_AUDIO_DIR = "testowy_folder_nagrania"
OUTPUT_DIR = "./models_output"

SAMPLE_RATE = 44100      
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40

# --- Konfiguracja Portu i Nagrania ---
PORT = "COM4" 
BAUD = 1000000
RECORD_SECONDS = 5
NUM_SAMPLES = SAMPLE_RATE * RECORD_SECONDS 
ADC_BITS = 14          # rozdzielczość ADC w STM32



# Ścieżki modelu

model_path = os.path.join(OUTPUT_DIR, "animal_sound_model.tflite")
scaler_path = os.path.join(OUTPUT_DIR, "scaler.joblib")
le_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")


# Walidacja plików i ładowanie

def load_ml_assets():
    """Ładowanie modelu, scalera i enkodera, jeśli istnieją."""
    try:
        if not all(os.path.exists(p) for p in [model_path, scaler_path, le_path]):
             print("[ERROR] Brakuje plików modelu/scalera/encodera. Zakończenie.")
             sys.exit(1)
             
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        scaler = joblib.load(scaler_path)
        le = joblib.load(le_path)
        return interpreter, scaler, le
    except Exception as e:
        print(f"[ERROR] Błąd ładowania zasobów ML: {e}")
        sys.exit(1)

interpreter, scaler, le = load_ml_assets()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


# Funkcje narzędziowe 


def extract_features(path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, sr=SAMPLE_RATE):
    """Ekstrakcja cech (MFCC) z pliku audio."""
    try:
        y, sr = sf.read(path)
        y = y.astype(np.float32)
        
        # Ramkowanie + okno Hamming
        frames = []
        window = get_window("hamming", n_fft, fftbins=True)
        for start in range(0, len(y) - n_fft + 1, hop_length):
            frame = y[start:start+n_fft] * window
            frames.append(frame)
        frames = np.array(frames)

        # FFT + magnituda
        mag = np.abs(rfft(frames, n=n_fft))

        # Mel filter bank
        mel_fb = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
        mel_energy = np.log(np.maximum(mag @ mel_fb.T, 1e-12))

        # DCT → MFCC
        mfcc = dct(mel_energy, type=2, axis=1, norm='ortho')[:, :n_mfcc]

        # średnia + std
        features = np.hstack([mfcc.mean(axis=0), mfcc.std(axis=0)])
        return features
    except Exception as e:
        print(f"Błąd ekstrakcji cech: {e}")
        return None

def save_wav(filename, samples):
    """Zapisuje próbki do pliku WAV."""
    try:
        with wave.open(filename, 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            wav_file.writeframes(samples.tobytes())
        print(f"Zapisano plik WAV: {filename}")
    except wave.Error as e:
        print(f"Błąd podczas zapisu pliku WAV {filename}: {e}")

def trim_wav_to_5s(filename):
    """Przycinanie pliku WAV do maksymalnie 5 sekund (na podstawie RECORD_SECONDS)."""
    try:
        with wave.open(filename, 'r') as wav_file:
            params = wav_file.getparams()
            n_frames = wav_file.getnframes()
            frames = wav_file.readframes(n_frames)

        max_frames = SAMPLE_RATE * RECORD_SECONDS
        if n_frames > max_frames:
            frames = frames[:max_frames * params.sampwidth] 
            
            # Tworzymy nowy plik z przyciętymi ramkami
            with wave.open(filename, 'w') as wav_file_out:
                wav_file_out.setparams(params)
                wav_file_out.setnframes(max_frames)
                wav_file_out.writeframes(frames)
            print(f"Plik {filename} przycięty do {RECORD_SECONDS} sekund.")
        else:
             print(f"Plik {filename} ma {n_frames/SAMPLE_RATE:.2f}s i nie wymaga przycinania.")

    except wave.Error as e:
        print(f"Błąd podczas przycinania pliku WAV {filename}: {e}")

# Klasa Robocza

class SerialWorker(QThread):
    
    # Sygnały do komunikacji z wątkiem głównym
    data_ready = pyqtSignal(bytearray)
    progress_update = pyqtSignal(int, int) # (aktualne_bajty, oczekiwane_bajty)
    error_signal = pyqtSignal(str)

    def __init__(self, port_name, baud_rate, target_bytes, ser_instance, parent=None):
        super().__init__(parent)
        self.port_name = port_name
        self.baud_rate = baud_rate
        self.target_bytes = target_bytes
        self.ser = ser_instance # Przekazana instancja portu
        
    def run(self):
        
        data = bytearray()
        
        # Sprawdzamy, czy port jest otwarty
        if not self.ser or not self.ser.is_open:
            self.error_signal.emit(f"Błąd: Port szeregowy ({self.port_name}) nie jest otwarty w wątku roboczym.")
            return

        try:
            while len(data) < self.target_bytes:
                
                chunk = self.ser.read(1000) 
                
                if chunk:
                    data.extend(chunk)
                    
                    self.progress_update.emit(len(data), self.target_bytes)
                else:
                    break
            
            self.data_ready.emit(data)

        except serial.SerialException as e:
            self.error_signal.emit(f"Błąd odczytu z portu {self.port_name}: {e}")
        except Exception as e:
            self.error_signal.emit(f"Nieoczekiwany błąd w wątku roboczym: {e}")


# Klasa GUI

class MyApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ser = None # Instancja portu szeregowego
        self.folder_path = "D:\Machine learning projects\Projekt dyplomowy\Panel\Odebrane nagrania" # Domyślny folder do zapisu
        self.file_wav_name = "Output.wav"
        # Podpięcie przycisków
        self.ui.Test_all_from_folder_button.clicked.connect(self.run_inference_all)
        self.ui.Select_file_for_test.clicked.connect(self.select_audio_file)
        self.ui.Pick_up_samples_button.clicked.connect(self.start_data_pickup) # Zmienione na start_data_pickup
        self.ui.Serial_port_open_button.stateChanged.connect(self.Open_serial_port)
        self.ui.Select_folder_for_write_wav_button.clicked.connect(self.select_output_folder)
        self.ui.File_wav_name.setText(self.file_wav_name)
        self.ui.File_wav_name.textChanged.connect(self.update_wav_filename)
        self.ui.Select_file_for_compare_with_GPT.clicked.connect(self.select_audio_file_for_GPT)
        # Wątek roboczy (uchwyt)
        self.worker_thread = None


    def predict_file(self, filepath):
        
        features = extract_features(filepath)
        if features is None:
            return f"[WARN] Błąd podczas przetwarzania pliku: {filepath}\n"

        # Skalowanie i predykcja
        scaled_features = scaler.transform(features.reshape(1, -1)).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], scaled_features)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        predicted_class_index = int(np.argmax(predictions))
        confidence = float(predictions[0][predicted_class_index])
        predicted_label = le.inverse_transform([predicted_class_index])[0]

        result = (
            f"--- Plik: {os.path.basename(filepath)} ---\n"
            f"Przewidziana kategoria: {predicted_label}\n"
            f"Pewność: {confidence:.4f}\n"
        )
        return result
    
    def predict_file_GPT(self, filepath):
        client = OpenAI(api_key=API_KEY_GPT)
        with open(filepath, "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode("utf-8")
        lista_zwierzat = "cat, cow, crow, dog, frog, hen, insects, pig, rooster, sheep, horse"

        try:
            
            response = client.chat.completions.create(
                model="gpt-4o-audio-preview",
                modalities=["text"], # Chcemy odpowiedź tekstową
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"Listen to this audio and tell me which animal from this list it is: {lista_zwierzat}. Return only the name of the animal."},
                            {"type": "input_audio", "input_audio": {"data": encoded_audio, "format": "wav"}}
                        ]
                    }
                ]
            )
        except Exception as e:
                return f"ERROR: {e}"
        # 5. Zapisanie odpowiedzi do zmiennej
        predict_result = response.choices[0].message.content.strip().lower()
        result = (
            f"--- Plik: {os.path.basename(filepath)} ---\n"
            f"Przewidziana kategoria przez GPT: {predict_result}\n"
        )
        return result
    
    def update_wav_filename(self, text):
        """Aktualizacja nazwy pliku WAV na podstawie wpisanego tekstu"""
        self.file_wav_name = text# if text.endswith('.wav') else text + '.wav'
       
        
    def select_output_folder(self):
        """Wybór folderu do zapisu nagrania"""
        folder_path = QFileDialog.getExistingDirectory(self, "Wybierz folder do zapisu nagrania", "")
        if folder_path:
            global OUTPUT_DIR
            OUTPUT_DIR = folder_path
            self.ui.Text_window.append(f"Wybrano folder do zapisu nagrania: {OUTPUT_DIR}\n")
            self.folder_path = folder_path
        return self.folder_path
    
    
    def run_inference_all(self):
        """Analiza wszystkich nagrań z folderu"""
        self.ui.Text_window.clear()
        audio_files = [f for f in os.listdir(TEST_AUDIO_DIR) if f.lower().endswith(('.wav', '.flac'))]

        if not audio_files:
            self.ui.Text_window.append(f"[INFO] Folder '{TEST_AUDIO_DIR}' jest pusty lub brak plików .wav/.flac.")
            return

        for filename in audio_files:
            filepath = os.path.join(TEST_AUDIO_DIR, filename)
            result = self.predict_file(filepath)
            self.ui.Text_window.append(result)

        self.ui.Text_window.append("[DONE] Zakończono testowanie.\n")

    def select_audio_file(self):
        """Wybór pojedynczego pliku i analiza"""
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Wybierz plik audio",
            filter="Pliki audio (*.wav *.flac *.mp3)"
        )
        if file_path:
            #self.ui.Text_window.clear()
            self.ui.Text_window.append(f"Wybrano plik: {file_path}\n")
            result = self.predict_file(file_path)
            self.ui.Text_window.append(result)

    def select_audio_file_for_GPT(self):
        """Wybór pojedynczego pliku i analiza GPT"""
        file_path, _ = QFileDialog.getOpenFileName(
            parent=self,
            caption="Wybierz plik audio",
            filter="Pliki audio (*.wav *.flac *.mp3)"
        )
        if file_path:
            #self.ui.Text_window.clear()
            self.ui.Text_window.append(f"Wybrano plik: {file_path} do analizy GPT\n")
            result = self.predict_file_GPT(file_path)
            self.ui.Text_window.append(result)
            
    def Open_serial_port(self, state):
        """Otwieranie/zamykanie portu szeregowego"""
        if state: # Przycisk włączony (otwieramy)
            try:
                # Ważne: ustawienie timeout=1 pozwala pętli odczytu w wątku na nieblokujące sprawdzanie
                self.ser = serial.Serial(PORT, BAUD, timeout=10) 
                self.ui.Text_window.append(f"Port {PORT} otwarty z prędkością {BAUD} bps.\n")
            except serial.SerialException as e:
                self.ui.Text_window.append(f"Błąd otwarcia portu {PORT}: {e}\n")
                self.ui.Serial_port_open_button.setChecked(False) # Zamykamy przycisk
                self.ser = None
        else: # Przycisk wyłączony (zamykamy)
            if self.ser and self.ser.is_open:
                self.ser.close()
                self.ui.Text_window.append(f"Port {PORT} zamknięty.\n")
            self.ser = None

    # --- Zrefaktoryzowana obsługa odbioru danych (rozbita na funkcje) ---

    def start_data_pickup(self):
        """Inicjuje odbiór danych w osobnym wątku."""
        if not self.ser or not self.ser.is_open:
            self.ui.Text_window.append("Port szeregowy nie jest otwarty. Otwórz go przed odbiorem danych.\n")
            return
            
        if self.worker_thread and self.worker_thread.isRunning():
            self.ui.Text_window.append("[WARN] Odbiór danych już trwa. Poczekaj na zakończenie.\n")
            return

        target_bytes = NUM_SAMPLES * 2 
        self.ui.Text_window.append(f"Czekam na {target_bytes} bajtów od STM32...\n")
        
        # Tworzenie i uruchamianie wątku roboczego
        self.worker_thread = SerialWorker(PORT, BAUD, target_bytes, self.ser)
        
        # Podpinanie sygnałów do slotów w głównym wątku
        self.worker_thread.progress_update.connect(self.update_progress_ui)
        self.worker_thread.data_ready.connect(self.process_received_data)
        self.worker_thread.error_signal.connect(self.handle_worker_error)
        self.worker_thread.finished.connect(self.handle_worker_finished)
        
        self.worker_thread.start() # Uruchomienie wątku

    def update_progress_ui(self, current_bytes, target_bytes):
        """Aktualizuje GUI informacją o postępie (uruchamiane w wątku głównym)."""
        # Używamy kursora do manipulacji tekstem, aby wiarygodnie nadpisywać linię postępu.
        cursor = self.ui.Text_window.textCursor()
        cursor.movePosition(cursor.End)
        
        # Znajdź początek ostatniej linii
        cursor.movePosition(cursor.StartOfLine, cursor.KeepAnchor)
        last_line = cursor.selectedText()
        
        # Jeśli ostatnia linia zawiera "Odebrano", usuń ją (to jest poprzednia linia postępu)
        if last_line.startswith("Odebrano"):
            cursor.removeSelectedText()
            cursor.movePosition(cursor.End)
        
        # Wstaw nową linię postępu z \r
        text = f"Odebrano {current_bytes}/{target_bytes} bajtów\r"
        cursor.insertText(text)
        
        # Upewniamy się, że kursor jest widoczny
        self.ui.Text_window.ensureCursorVisible()

    def handle_worker_error(self, error_message):
        """Obsługa błędów z wątku roboczego."""
        self.ui.Text_window.append(f"\n[ERROR Z WĄTKU] {error_message}")
        
    def handle_worker_finished(self):
        """Wywoływana po zakończeniu pracy przez wątek."""
        # Wątek zakończył pracę, nawet jeśli zakończył się błędem
        self.ui.Text_window.append("\n[INFO] Wątek odbioru danych zakończył działanie.")
        # Opcjonalnie: zamykamy port, jeśli był otwarty
 

    def process_received_data(self, data):
        target_bytes = NUM_SAMPLES * 2
        
        self.ui.Text_window.append(f"Odebrano łącznie {len(data)} bajtów ({len(data)//2} próbek)\n")

        # --- Uzupełnianie zerami w razie potrzeby ---
        if len(data) < target_bytes:
            missing_bytes = target_bytes - len(data)
            self.ui.Text_window.append(f"Uwaga: brakuje {missing_bytes} bajtów. Uzupełniam zerami.\n")
            data.extend(b'\x00' * missing_bytes)
        
        # --- Konwersja do próbek 16-bit (Little-endian) ---
        samples = np.frombuffer(data, dtype='<u2')

        # --- Zamiana na signed wokół zera ---
        mid = 2**(ADC_BITS-1) # Poprawione obliczenie środka
        samples_signed = samples.astype(np.int32) - mid

        # --- Skalowanie do pełnych 16 bitów ---
        samples_int16 = samples_signed.astype(np.int16) 
        
        # --- Wariant 1: RAW ---
        raw = samples_int16.copy()

        # --- Wariant 2: ZEROED ---
        # Logika zerowania skrajnych wartości (niezgodnych z 14-bitowym zakresem)
        zeroed = samples_int16.copy()
        zeroed[(zeroed == 32767) | (zeroed == -32768)] = 0

        # --- Wariant 3: AVG (średnia ruchoma) ---
        window = 5
        kernel = np.ones(window) / window
        avg = np.convolve(samples_int16, kernel, mode='same').astype(np.int16)
        # Nazwy plików
        OUTPUT_RAW = os.path.join(self.folder_path, self.file_wav_name + "_raw" + ".wav")
        OUTPUT_XLSX = os.path.join(self.folder_path, self.file_wav_name + "_samples.xlsx")
        # Zapis plików
        save_wav(OUTPUT_RAW, raw)
        
        # Przycięcie
        trim_wav_to_5s(OUTPUT_RAW)        
        # Po odebraniu i przetworzeniu danych zamykamy port, jeśli jest otwarty
        if self.ser and self.ser.is_open:
            self.ser.close()
            self.ui.Text_window.append(f"Port {PORT} automatycznie zamknięty po odbiorze.\n")
            self.ui.Serial_port_open_button.setChecked(False) # Zmieniamy stan przycisku


# -------------------------
# Start aplikacji
# -------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
