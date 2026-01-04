import os
import numpy as np
import librosa
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf
from tensorflow import keras
import argparse
from scipy.signal import get_window
from scipy.fft import rfft
from scipy.fftpack import dct
import numpy as np
import soundfile as sf
    
#argumenty z linii polecen

parser = argparse.ArgumentParser(description="Trening modelu rozpoznawania dźwięków zwierząt.")
parser.add_argument("--data_path", type=str, default="./Do nauki nagrania",
                    help="Ścieżka do folderu z nagraniami do nauki.")
parser.add_argument("--output_dir", type=str, default="./models_output", help="Folder do zapisu modeli.")
parser.add_argument("--epochs", type=int, default=400, help="Liczba epok treningu.")
parser.add_argument("--batch_size", type=int, default=32, help="Rozmiar batcha.")
args = parser.parse_args()

AUDIO_DIR = args.data_path
OUTPUT_DIR = args.output_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
SAMPLE_RATE = 44100
 
N_MFCC = 40
N_FFT = 1024
HOP_LENGTH = 512
N_MELS = 40

# Ekstrakcja cech

def extract_features(path, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, sr=SAMPLE_RATE):
    
    try:
        y, sr = sf.read(path)
        y = y.astype(np.float32)

        # Konwersja stereo -> mono
        if y.ndim == 2:
            y = y.mean(axis=1)

        # Ramkowanie + Hamming
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

        # DCT -> MFCC
        mfcc = dct(mel_energy, type=2, axis=1, norm='ortho')[:, :n_mfcc]

        # średnia + std
        features = np.hstack([mfcc.mean(axis=0), mfcc.std(axis=0)])
        return features

    except Exception as e:
        print(f"[WARN] Błąd podczas przetwarzania pliku {path}: {e}")
        return None

#wczytanie danych z folderu
X_list, y_list = [], []
errors = 0

for fname in os.listdir(AUDIO_DIR):
    if not fname.lower().endswith(".wav"):
        continue
    try:
        label = fname.split("_")[0].lower()
        path = os.path.join(AUDIO_DIR, fname)
        feats = extract_features(path)
        if feats is not None:
            X_list.append(feats)
            y_list.append(label)
    except Exception as e:
        print(f"[WARN] Problem z {fname}: {e}")
        errors += 1

X = np.array(X_list)
y = np.array(y_list)
print(f"[INFO] Zbudowano macierz cech: {X.shape}, etykiety: {len(y)} (błędy: {errors})")
print(f"[INFO] Klasy wykryte w zbiorze: {np.unique(y)}")


le = LabelEncoder()
y_enc = le.fit_transform(y)

# podzial danych
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_enc,
    test_size=0.2,
    stratify=y_enc,
    random_state=RANDOM_STATE
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.25,          
    stratify=y_train_full,
    random_state=RANDOM_STATE
)

#skalowanie
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

joblib.dump(le, os.path.join(OUTPUT_DIR, "label_encoder.joblib"))
joblib.dump(scaler, os.path.join(OUTPUT_DIR, "scaler.joblib"))
print("[INFO] Zapisano label encoder i scaler.")


num_classes = len(np.unique(y_enc))
input_dim = X_train.shape[1]

model = keras.Sequential([
    keras.layers.Input(shape=(input_dim,)),

    keras.layers.Dense(256, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(128, activation="relu"),
    keras.layers.BatchNormalization(),
    keras.layers.Dropout(0.3),

    keras.layers.Dense(64, activation="relu"),
    keras.layers.Dropout(0.2),  

    keras.layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

#trening z callbackami
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=20,
        restore_best_weights=True,
        verbose=1
    ),

    keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(OUTPUT_DIR, "best_model.h5"),
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=10,
        verbose=1,
        min_lr=1e-6
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=args.epochs,
    batch_size=args.batch_size,
    verbose=2,
    callbacks=callbacks
)
# Ewaluacja
loss, acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Test loss: {loss:.4f}, Test accuracy: {acc:.4f}")

best_model = keras.models.load_model(os.path.join(OUTPUT_DIR, "best_model.h5"))
best_loss, best_acc = best_model.evaluate(X_test, y_test, verbose=0)
print(f"[RESULT] Najlepszy model -> Test loss: {best_loss:.4f}, Test accuracy: {best_acc:.4f}")

keras_model_path = os.path.join(OUTPUT_DIR, "animal_sound_model.h5")
best_model.save(keras_model_path)
print(f"[INFO] Zapisano finalny model Keras: {keras_model_path}")

#Eksport do TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
tflite_model = converter.convert()
tflite_path = os.path.join(OUTPUT_DIR, "animal_sound_model.tflite")
with open(tflite_path, "wb") as f:
    f.write(tflite_model)
print(f"[INFO] Zapisano TFLite model (float32): {tflite_path}")

try:
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_dynamic = converter.convert()
    tflite_dynamic_path = os.path.join(OUTPUT_DIR, "animal_sound_model_dynamic.tflite")
    with open(tflite_dynamic_path, "wb") as f:
        f.write(tflite_dynamic)
    print(f"[INFO] Zapisano model z dynamiczną kwantyzacją: {tflite_dynamic_path}")
except Exception as e:
    print(f"[WARN] Dynamiczna kwantyzacja nie powiodła się: {e}")

print("[DONE] Wszystkie pliki zapisane w:", OUTPUT_DIR)

#Generowanie plikow do miktrokontrolera
scaler = joblib.load(os.path.join(OUTPUT_DIR, "scaler.joblib"))
le = joblib.load(os.path.join(OUTPUT_DIR, "label_encoder.joblib"))

mean = scaler.mean_.astype(np.float32)
scale = scaler.scale_.astype(np.float32)
labels = list(le.classes_)

Q_BITS = 14
Q_MAX = 2**(Q_BITS-1)-1

X_train_scaled = scaler.transform(X_train)
X_train_clipped = np.clip(X_train_scaled, -1.0, 1.0)
X_train_q14 = (X_train_clipped / 3.0 * Q_MAX).astype(np.int16)

def fmt_array(name, arr, per_line=8):
    s = f"static const float {name}[] = {{\n"
    for i in range(0, len(arr), per_line):
        s += "    " + ", ".join(f"{v:.8f}f" for v in arr[i:i+per_line]) + ",\n"
    s += "};\n"
    return s

def fmt_int_array(name, arr, per_line=8):
    s = f"static const int16_t {name}[] = {{\n"
    for i in range(0, len(arr), per_line):
        s += "    " + ", ".join(str(v) for v in arr[i:i+per_line]) + ",\n"
    s += "};\n"
    return s

with open(os.path.join(OUTPUT_DIR, "scaler_params.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("// Autogenerated by Python script\n\n")
    f.write(f"#define FEATURE_COUNT {X_train.shape[1]}\n\n")
    f.write(fmt_array("SCALER_MEAN", mean))
    f.write("\n")
    f.write(fmt_array("SCALER_SCALE", scale))
    f.write("\n")
    f.write(fmt_int_array("SCALER_Q14_MEAN", X_train_q14.mean(axis=0).astype(np.int16)))
    f.write("\n")
    f.write(f"static const int SCALER_LEN = {len(mean)};\n\n")
    f.write("static const char *LABELS[] = {\n")
    for lab in labels:
        f.write(f"    \"{lab}\",\n")
    f.write("};\n")
    f.write(f"static const int NUM_LABELS = {len(labels)};\n")

print("Wygenerowano scaler_params.h (Q14 ADC)")

#mel mel filter bank, DCT matrix, Hamming window
FRAME_SIZE = 1024
HOP_LENGTH = 512
sr = SAMPLE_RATE
n_fft = FRAME_SIZE
n_mels = 40
n_mfcc = 40

mel_filter = librosa.filters.mel(
    sr=sr, n_fft=n_fft, n_mels=n_mels,
    fmin=0, fmax=sr/2
).astype(np.float32)

def fmt_matrix(name, mat):
    s = f"static const float {name}[{mat.shape[0]}][{mat.shape[1]}] = {{\n"
    for row in mat:
        s += "    {" + ", ".join(f"{v:.8e}f" for v in row) + "},\n"
    s += "};\n"
    return s

with open(os.path.join(OUTPUT_DIR, "mel_filter_banks.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("// Autogenerated Mel filter banks\n\n")
    f.write(f"#define NUM_MEL_FILTERS {n_mels}\n")
    f.write(f"#define FFT_BINS {n_fft//2+1}\n\n")
    f.write(fmt_matrix("mel_filter_banks", mel_filter))

print("Wygenerowano mel_filter_banks.h")

dct_matrix_manual = np.zeros((n_mfcc, n_mels), dtype=np.float32)
for i in range(n_mfcc):
    dct_matrix_manual[i, :] = np.cos(np.pi * i / n_mels * (np.arange(n_mels) + 0.5))

dct_matrix_manual[0, :] *= 1.0 / np.sqrt(n_mels)
dct_matrix_manual[1:, :] *= np.sqrt(2.0 / n_mels)

with open(os.path.join(OUTPUT_DIR, "dct_matrix.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("// Autogenerated DCT matrix\n\n")
    f.write(f"#define NUM_MFCC_COEFFS {n_mfcc}\n")
    f.write(f"#define NUM_MEL_FILTERS {n_mels}\n\n")
    f.write(fmt_matrix("dct_matrix", dct_matrix_manual))

print("Wygenerowano dct_matrix.h")

hamming = np.hamming(FRAME_SIZE).astype(np.float32)

with open(os.path.join(OUTPUT_DIR, "hamming_window.h"), "w") as f:
    f.write("#pragma once\n\n")
    f.write("// Autogenerated Hamming window\n\n")
    f.write(f"#define FRAME_SIZE {FRAME_SIZE}\n\n")
    f.write(fmt_array("hamming_window", hamming))

print("Wygenerowano hamming_window.h")
