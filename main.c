/* USER CODE BEGIN Header */
/**
 ******************************************************************************
 * @file           : main.c
 * @brief          : Main program body
 ******************************************************************************
 * @attention
 *
 * Copyright (c) 2025 STMicroelectronics.
 * All rights reserved.
 *
 * This software is licensed under terms that can be found in the LICENSE file
 * in the root directory of this software component.
 * If no LICENSE file comes with this software, it is provided AS-IS.
 *
 ******************************************************************************
 */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"
#include "app_x-cube-ai.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "ssd1306_fonts.h"
#include "ssd1306.h"
#include "ml_network.h"
#include "ml_network_data.h"
#include "ai_datatypes_defines.h"
#include "ai_platform.h"

#include <string.h>
#include <stdio.h>
#include <math.h>
#include <stdbool.h>
#include "stm_status.h"
#include "ml_network_data_params.h"

#include "scaler_params.h"
#include "mel_filter_banks.h"
#include "dct_matrix.h"
#include "hamming_window.h"
#include "arm_math.h"
#include "stdlib.h"



/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
#define NUM_LABELS 10
float32_t magnitude_buffer[FRAME_SIZE / 2 + 1];
volatile uint16_t counter;
volatile static uint16_t audio_buf[NUM_SAMPLES]; // 16-bit próbki (mono, left)
volatile uint8_t recording_flag = 0;
uint16_t toReceive;
volatile uint8_t uart_tx_complete = 1;
ai_handle ai_network = AI_HANDLE_NULL; // globalny uchwyt do sieci
float32_t features[FEATURE_COUNT];
//AI_ALIGNED(32) static uint8_t ai_activations_buffer[AI_ML_NETWORK_DATA_ACTIVATION_SIZE];

volatile App_State app_states = State_idle;
static uint32_t send_index = 0;   // globalny indeks do wysyłki audio
#define CHUNK_SIZE 32000U

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
static arm_rfft_fast_instance_f32 fft_instance;

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
ADC_HandleTypeDef hadc1;

I2C_HandleTypeDef hi2c1;

RAMECC_HandleTypeDef hramecc1_m1;
RAMECC_HandleTypeDef hramecc1_m2;
RAMECC_HandleTypeDef hramecc1_m3;
RAMECC_HandleTypeDef hramecc1_m4;
RAMECC_HandleTypeDef hramecc1_m5;
RAMECC_HandleTypeDef hramecc2_m1;
RAMECC_HandleTypeDef hramecc2_m2;
RAMECC_HandleTypeDef hramecc2_m3;
RAMECC_HandleTypeDef hramecc2_m4;
RAMECC_HandleTypeDef hramecc2_m5;
RAMECC_HandleTypeDef hramecc3_m1;
RAMECC_HandleTypeDef hramecc3_m2;

TIM_HandleTypeDef htim1;

UART_HandleTypeDef huart1;
UART_HandleTypeDef huart3;
DMA_HandleTypeDef hdma_usart1_tx;

/* USER CODE BEGIN PV */
void normalize_features(float* features, int count);
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_DMA_Init(void);
static void MX_ADC1_Init(void);
static void MX_USART1_UART_Init(void);
static void MX_TIM1_Init(void);
static void MX_RAMECC_Init(void);
static void MX_I2C1_Init(void);
/* USER CODE BEGIN PFP */

static void adc_to_int16_in_place(uint16_t* audio_buf, int len);

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */

static arm_rfft_fast_instance_f32 fft_instance;

void normalize_features(float* features, int count) {
	for (int i = 0; i < count; i++) {
		features[i] = (features[i] - SCALER_MEAN[i]) / SCALER_SCALE[i];
	}
}
void debug_msg(const char *msg)
{
	HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);
}
AI_ALIGNED(32) static uint8_t ai_activations_buffer[AI_ML_NETWORK_DATA_ACTIVATIONS_SIZE];
AI_ALIGNED(32) static ai_float out_data[NUM_LABELS];

void init_network(void) {
	ai_error err;
	char msg[120];

	const void* weights = ai_ml_network_data_weights_get();
	if (!weights) {
		debug_msg("Error: Weights pointer is NULL\r\n");
		Error_Handler();
	}

	ai_network_params params = AI_NETWORK_PARAMS_INIT(
			AI_ML_NETWORK_DATA_WEIGHTS(weights),
			AI_ML_NETWORK_DATA_ACTIVATIONS(ai_activations_buffer)
	);

	err = ai_ml_network_create(&ai_network, AI_ML_NETWORK_DATA_CONFIG);
	if (err.type != AI_ERROR_NONE) {
		snprintf(msg, sizeof(msg),
				"AI network creation failed: type=%lu, code=%lu\r\n",
				(unsigned long)err.type, (unsigned long)err.code);
		HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);
		Error_Handler();
	}

	if (!ai_ml_network_init(ai_network, &params)) {
		err = ai_ml_network_get_error(ai_network);
		snprintf(msg, sizeof(msg),
				"AI network initialization failed: type=%lu, code=%lu\r\n",
				(unsigned long)err.type, (unsigned long)err.code);
		HAL_UART_Transmit(&huart1, (uint8_t*)msg, strlen(msg), HAL_MAX_DELAY);
		Error_Handler();
	}

	debug_msg("AI network initialized successfully\r\n");
}





static void adc_to_int16_in_place(uint16_t* audio_buf, int len) {

	int32_t mid = 1 << 13; // 14-bit ADC
	float scale = 32767.0f / mid;

	char msg[128];
	debug_msg("Start adc_to_int16_in_place\r\n");

	for (int i = 0; i < len; i++) {
		// Read the 14-bit ADC value from the buffer
		int32_t adc_val = (int32_t)audio_buf[i];

		// Convert the value
		int32_t val = adc_val - mid;
		int32_t scaled = (int32_t)(val * scale);

		// Clamp the value to the int16_t range
		if (scaled > 32767) scaled = 32767;
		if (scaled < -32768) scaled = -32768;

		// Overwrite the original uint16_t value with the new int16_t value
		audio_buf[i] = (int16_t)scaled;

		// Debug for the first 10 samples
		if (i < 10) {
			snprintf(msg, sizeof(msg), "ADC[%d]=%lu -> int16=%d\r\n", i, adc_val, (int16_t)audio_buf[i]);
			debug_msg(msg);
		}
	}

	debug_msg("adc_to_int16_in_place DONE\r\n");
}



// z prem faza
void compute_mfcc_features(const uint16_t* audio_buf, float* feature_vector, int len_samples) {
	char msg[128];
	debug_msg("Start compute_mfcc_features\r\n");

	// Konwersja ADC na int16 (w miejscu)
	adc_to_int16_in_place((uint16_t*)audio_buf, len_samples);
	debug_msg("ADC -> int16 conversion done\r\n");

	int num_frames = (len_samples - N_FFT) / HOP_LENGTH + 1;
	snprintf(msg, sizeof(msg), "Number of frames: %d\r\n", num_frames);
	debug_msg(msg);

	float fft_input[N_FFT];
	float fft_output[N_FFT];
	static arm_rfft_fast_instance_f32 S;
	static int fft_init = 0;

	if(!fft_init) {
		if(arm_rfft_fast_init_f32(&S, N_FFT) != ARM_MATH_SUCCESS) {
			debug_msg("FFT init failed!\r\n");
			return;
		}
		fft_init = 1;
		debug_msg("FFT initialized\r\n");
	}

	float mfcc_acc[N_MFCC*2] = {0}; // Sumy i sumy kwadratów
	//float last_sample = 0.0f; // Poprzednia próbka do preemfazy

	for(int frame = 0; frame < num_frames; frame++) {
		// Przetwarzanie ramki z preemfazą i okienkowaniem
		for(int i = 0; i < N_FFT; i++) {
			int index = frame * HOP_LENGTH + i;
			if (index >= len_samples) {
				fft_input[i] = 0.0f;
			} else {
				/*float sample = (float)((int16_t*)audio_buf)[index] / 32768.0f;
				float preemph_sample = (index == 0) ? sample : sample - 0.97f * last_sample;
				last_sample = sample; // Zapamiętanie dla następnej iteracji
				fft_input[i] = preemph_sample * hamming_window[i];

				tutaj z prem - emfaza*/

				float sample = (float)((int16_t*)audio_buf)[index] / 32768.0f;
				fft_input[i] = sample * hamming_window[i];
			}
		}

		// FFT
		arm_rfft_fast_f32(&S, fft_input, fft_output, 0);

		// Obliczanie magnitudy
		float mag[N_FFT/2+1];
		mag[0] = fabsf(fft_output[0]);
		for(int i = 1; i < N_FFT/2; i++) {
			mag[i] = sqrtf(fft_output[2*i]*fft_output[2*i] + fft_output[2*i+1]*fft_output[2*i+1]);
		}
		mag[N_FFT/2] = fabsf(fft_output[1]);

		// Filtr Mel
		float mel_energies[N_MELS];
		for(int m = 0; m < N_MELS; m++) {
			mel_energies[m] = 0.0f;
			for(int k = 0; k < N_FFT/2+1; k++) {
				mel_energies[m] += mag[k] * mel_filter_banks[m][k];
			}
			mel_energies[m] = logf(mel_energies[m] + 1e-12f);
		}

		// DCT -> MFCC
		float mfcc_frame[N_MFCC];
		for(int i = 0; i < N_MFCC; i++) {
			mfcc_frame[i] = 0.0f;
			for(int j = 0; j < N_MELS; j++) {
				mfcc_frame[i] += dct_matrix[i][j] * mel_energies[j];
			}
		}

		// Akumulacja statystyk
		for(int i = 0; i < N_MFCC; i++) {
			mfcc_acc[i] += mfcc_frame[i];
			mfcc_acc[i+N_MFCC] += mfcc_frame[i] * mfcc_frame[i];
		}
	}

	// Obliczanie średniej i odchylenia standardowego
	for(int i = 0; i < N_MFCC; i++) {
		float mean = mfcc_acc[i] / num_frames;
		float std = sqrtf(fmaxf(0.0f, (mfcc_acc[i+N_MFCC]/num_frames) - mean*mean));
		feature_vector[i] = mean;
		feature_vector[i+N_MFCC] = std;
	}
	debug_msg("compute_mfcc_features DONE\r\n");
}
void run_inference(volatile uint16_t *audio_buf, int num_samples)
{
	char msg[128];
	debug_msg("run_inference...\r\n");

	// Debug: pierwsze 10 próbek surowego ADC
	snprintf(msg, sizeof(msg), "Raw value ADC: ");
	debug_msg(msg);
	for(int i=0; i<10 && i<num_samples; i++){
		snprintf(msg, sizeof(msg), "%u ", audio_buf[i]);
		debug_msg(msg);
	}
	debug_msg("\r\n");

	// Obliczanie cech MFCC
	// Używamy zmiennej 'num_samples' zamiast stałej 'SAMPLE_RATE'
	// żeby funkcja była bardziej uniwersalna.
	compute_mfcc_features((const uint16_t*)audio_buf, features, num_samples);

	// Debug: pierwsze 10 cech po MFCC (przed normalizacją)
	snprintf(msg, sizeof(msg), "MFCC features - first 10 (before normalization): ");
	debug_msg(msg);
	for(int i=0; i<10 && i<FEATURE_COUNT; i++){
		snprintf(msg, sizeof(msg), "%.6f ", features[i]);
		debug_msg(msg);
	}
	debug_msg("\r\n");

	// 🔥 KLUCZOWA POPRAWKA: Normalizacja cech 🔥
	normalize_features(features, FEATURE_COUNT);

	// Debug: pierwsze 10 ZNORMALIZOWANYCH cech
	snprintf(msg, sizeof(msg), "MFCC features - first 10 (after normalization): ");
	debug_msg(msg);
	for(int i=0; i<10 && i<FEATURE_COUNT; i++){
		snprintf(msg, sizeof(msg), "%.6f ", features[i]);
		debug_msg(msg);
	}
	debug_msg("\r\n");

	// Wejście/wyjście AI
	debug_msg("Preparing in/out network...\r\n");
	ai_u16 n_input = 0, n_output = 0;
	ai_buffer* ai_input = ai_ml_network_inputs_get(ai_network, &n_input);
	ai_buffer* ai_output = ai_ml_network_outputs_get(ai_network, &n_output);

	if(n_input < 1 || n_output < 1) {
		debug_msg("Error\r\n");
		return;
	}

	// Debug informacji o sieci
	snprintf(msg, sizeof(msg), "Number of inputs: %u, input number: %u\r\n", n_input, n_output);
	debug_msg(msg);
	snprintf(msg, sizeof(msg), "size output: %lu, size: %lu\r\n",
			ai_input[0].size, ai_output[0].size);
	debug_msg(msg);

	// Przypisanie wskaźnika do ZNORMALIZOWANYCH cech
	ai_input[0].data = AI_HANDLE_PTR(features);
	ai_output[0].data = AI_HANDLE_PTR(out_data);

	// Uruchomienie sieci
	debug_msg("Network running...\r\n");
	ai_i32 result = ai_ml_network_run(ai_network, ai_input, ai_output);
	if(result != 1) {
		snprintf(msg, sizeof(msg), "Error during run network: %ld\r\n", result);
		debug_msg(msg);
		return;
	}
	debug_msg("Network run succesful\r\n");

	// Wybór klasy max
	int best_idx = 0;
	float best_val = out_data[0];

	debug_msg("Result of prediction:\r\n");
	for(int i = 0; i < NUM_LABELS; i++) {
		snprintf(msg, sizeof(msg), "Class %d (%s): %.2f%%\r\n", i, LABELS[i], out_data[i]*100.0f);
		debug_msg(msg);
		if(out_data[i] > best_val) {
			best_val = out_data[i];
			best_idx = i;
		}
	}

	snprintf(msg, sizeof(msg), "The highest propability: %s: %.2f%%\r\n", LABELS[best_idx], best_val*100.0f);
	debug_msg(msg);
	clear_ssd1306();
	ssd1306_SetCursor(0, 0);
	ssd1306_WriteString ((char*)LABELS[best_idx], Font_11x18, 1);
	ssd1306_UpdateScreen(); // Display
	HAL_Delay(10);
	char procenty[128];
	snprintf(procenty, sizeof(procenty),"%f", best_val*100.0f);
	ssd1306_SetCursor(0, 18);
	ssd1306_WriteString (procenty, Font_11x18, 1);
	HAL_Delay(10);
	ssd1306_UpdateScreen(); // Display
	debug_msg("run_inference finished.\r\n");
	HAL_Delay(2000);
}


void StartSendingChunks(void)
{
	send_index = 0;
	app_states = State_send;
}

void SendAllChunks_Blocking(void)
{
	send_index = 0;

	while (send_index < NUM_SAMPLES)
	{
		uint32_t remaining = NUM_SAMPLES - send_index;
		uint32_t chunk = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;

		HAL_StatusTypeDef status = HAL_UART_Transmit(
				&huart1,
				(uint8_t*)&audio_buf[send_index],
				chunk * sizeof(uint16_t),
				HAL_MAX_DELAY
		);

		if (status != HAL_OK) {
			// coś poszło nie tak, wyjdź
			Error_Handler();
		}

		send_index += chunk;
	}

	// Wszystko wysłane
	send_index = 0;
	uart_tx_complete = 1;
	app_states = State_wait_for_next_step;

}

void delay_us(uint16_t us)
{
	__HAL_TIM_SET_COUNTER(&htim1, 0);      // reset licznika
	while (__HAL_TIM_GET_COUNTER(&htim1) < us); // czekaj aż licznik osiągnie wartość
}


void StartRecording(void)
{
	uint32_t sample_count = 0;
	memset((void*)audio_buf, 0, NUM_SAMPLES * sizeof(uint16_t));
	while(sample_count < NUM_SAMPLES)
	{
		// Uruchomienie konwersji ADC
		if(HAL_ADC_Start(&hadc1) != HAL_OK)
		{
			Error_Handler();
		}
		// Oczekiwanie na zakończenie konwersji
		if(HAL_ADC_PollForConversion(&hadc1, HAL_MAX_DELAY) == HAL_OK)
		{
			// Odczytanie wartości i zapis do bufora
			audio_buf[sample_count] = HAL_ADC_GetValue(&hadc1);
		}
		else
		{
			// Obsługa błędu
			Error_Handler();
		}
		// Opóźnienie przed kolejną próbką
		sample_count++;
		delay_us(21); // test 44100 kHz

	}
	char msg[12] = "Recorded";
	HAL_ADC_Stop(&hadc1);
	clear_ssd1306();
    ssd1306_SetCursor(0, 0);
	ssd1306_WriteString ((char*)msg, Font_11x18, 1);
	ssd1306_UpdateScreen(); // Display
	app_states = State_wait_for_next_step;
}


void StopRecording(void)
{
	HAL_ADC_Stop_DMA(&hadc1);

}


/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{

  /* USER CODE BEGIN 1 */

  /* USER CODE END 1 */

  /* Enable the CPU Cache */

  /* Enable I-Cache---------------------------------------------------------*/
  SCB_EnableICache();

  /* Enable D-Cache---------------------------------------------------------*/
  SCB_EnableDCache();

  /* MCU Configuration--------------------------------------------------------*/

  /* Reset of all peripherals, Initializes the Flash interface and the Systick. */
  HAL_Init();

  /* USER CODE BEGIN Init */

  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */

  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_DMA_Init();
  MX_ADC1_Init();
  MX_USART1_UART_Init();
  MX_TIM1_Init();
  MX_RAMECC_Init();
  MX_I2C1_Init();
  MX_X_CUBE_AI_Init();
  /* USER CODE BEGIN 2 */

	counter = 0;
	SCB->CPACR |= ((3UL << 10*2)|(3UL << 11*2)); // włącz FPU
	init_network();
	debug_msg("FFT initialization\r\n");
	if (arm_rfft_fast_init_f32(&fft_instance, FRAME_SIZE) != ARM_MATH_SUCCESS) {
		debug_msg("FFT init failed!\r\n");
	}
	if (ssd1306_Init() == 1)
	{
		ssd1306_SetCursor(0,0);
		refreash_display("Initializated");
		HAL_Delay(500);
		debug_msg("SSD1306 success initalizated\r\n");

	}
	else
	{
		debug_msg("SSD1306 wasn't initalizated\r\n");
	}

	app_states = State_idle;

  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
	while (1)
	{
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */
		switch(app_states)
		{
		case State_idle:
			debug_msg("IDLE: Press button to start\r\n");
			refreash_display("IDLE");
			break;
		case State_recording:
			refreash_display("Recording");
			HAL_Delay(100);
			HAL_TIM_Base_Start(&htim1);   // start timera do delay_us
			debug_msg("Recording");
			StartRecording();
			debug_msg("Recording finished\r\n");
			break;
		case State_send:
			refreash_display("Sending");
			HAL_TIM_Base_Stop(&htim1); // Zatrzymaj timer
			SendAllChunks_Blocking();
			send_index =5;
			break;
		case State_AI_result:
			refreash_display("AI_result");
			run_inference(audio_buf,NUM_SAMPLES); // wysyłamy wynik predykcji
			app_states = State_idle;      // wracamy do stanu idle
			break;

		case State_recording_and_send_chunks:
			refreash_display("send_chunks");
			break;

		case State_wait_for_next_step:

			break;


		}
	}
  /* USER CODE END 3 */
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE1);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  __HAL_RCC_SYSCFG_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE0);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
  RCC_OscInitStruct.HSEState = RCC_HSE_BYPASS;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
  RCC_OscInitStruct.PLL.PLLM = 4;
  RCC_OscInitStruct.PLL.PLLN = 480;
  RCC_OscInitStruct.PLL.PLLP = 2;
  RCC_OscInitStruct.PLL.PLLQ = 20;
  RCC_OscInitStruct.PLL.PLLR = 2;
  RCC_OscInitStruct.PLL.PLLRGE = RCC_PLL1VCIRANGE_1;
  RCC_OscInitStruct.PLL.PLLVCOSEL = RCC_PLL1VCOWIDE;
  RCC_OscInitStruct.PLL.PLLFRACN = 0;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV2;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV2;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV2;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief ADC1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_ADC1_Init(void)
{

  /* USER CODE BEGIN ADC1_Init 0 */

  /* USER CODE END ADC1_Init 0 */

  ADC_MultiModeTypeDef multimode = {0};
  ADC_ChannelConfTypeDef sConfig = {0};

  /* USER CODE BEGIN ADC1_Init 1 */

  /* USER CODE END ADC1_Init 1 */

  /** Common config
  */
  hadc1.Instance = ADC1;
  hadc1.Init.ScanConvMode = ADC_SCAN_DISABLE;
  hadc1.Init.EOCSelection = ADC_EOC_SINGLE_CONV;
  hadc1.Init.LowPowerAutoWait = DISABLE;
  hadc1.Init.ContinuousConvMode = DISABLE;
  hadc1.Init.NbrOfConversion = 1;
  hadc1.Init.DiscontinuousConvMode = DISABLE;
  hadc1.Init.ExternalTrigConv = ADC_SOFTWARE_START;
  hadc1.Init.ExternalTrigConvEdge = ADC_EXTERNALTRIGCONVEDGE_NONE;
  hadc1.Init.ConversionDataManagement = ADC_CONVERSIONDATA_DR;
  hadc1.Init.Overrun = ADC_OVR_DATA_PRESERVED;
  hadc1.Init.LeftBitShift = ADC_LEFTBITSHIFT_NONE;
  hadc1.Init.OversamplingMode = DISABLE;
  hadc1.Init.Oversampling.Ratio = 1;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }
  hadc1.Init.ClockPrescaler = ADC_CLOCK_ASYNC_DIV1;
  hadc1.Init.Resolution = ADC_RESOLUTION_14B;
  if (HAL_ADC_Init(&hadc1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure the ADC multi-mode
  */
  multimode.Mode = ADC_MODE_INDEPENDENT;
  if (HAL_ADCEx_MultiModeConfigChannel(&hadc1, &multimode) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Regular Channel
  */
  sConfig.Channel = ADC_CHANNEL_18;
  sConfig.Rank = ADC_REGULAR_RANK_1;
  sConfig.SamplingTime = ADC_SAMPLETIME_1CYCLE_5;
  sConfig.SingleDiff = ADC_SINGLE_ENDED;
  sConfig.OffsetNumber = ADC_OFFSET_NONE;
  sConfig.Offset = 0;
  sConfig.OffsetSignedSaturation = DISABLE;
  if (HAL_ADC_ConfigChannel(&hadc1, &sConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN ADC1_Init 2 */

  /* USER CODE END ADC1_Init 2 */

}

/**
  * @brief I2C1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_I2C1_Init(void)
{

  /* USER CODE BEGIN I2C1_Init 0 */

  /* USER CODE END I2C1_Init 0 */

  /* USER CODE BEGIN I2C1_Init 1 */

  /* USER CODE END I2C1_Init 1 */
  hi2c1.Instance = I2C1;
  hi2c1.Init.Timing = 0x307075B1;
  hi2c1.Init.OwnAddress1 = 0;
  hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
  hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
  hi2c1.Init.OwnAddress2 = 0;
  hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
  hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
  hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;
  if (HAL_I2C_Init(&hi2c1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Analogue filter
  */
  if (HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_ENABLE) != HAL_OK)
  {
    Error_Handler();
  }

  /** Configure Digital filter
  */
  if (HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN I2C1_Init 2 */

  /* USER CODE END I2C1_Init 2 */

}

/**
  * @brief RAMECC Initialization Function
  * @param None
  * @retval None
  */
static void MX_RAMECC_Init(void)
{

  /* USER CODE BEGIN RAMECC_Init 0 */

  /* USER CODE END RAMECC_Init 0 */

  /* USER CODE BEGIN RAMECC_Init 1 */

  /* USER CODE END RAMECC_Init 1 */

  /** Initialize RAMECC1 M1 : AXI SRAM
  */
  hramecc1_m1.Instance = RAMECC1_Monitor1;
  if (HAL_RAMECC_Init(&hramecc1_m1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC1 M2 : ITCM-RAM
  */
  hramecc1_m2.Instance = RAMECC1_Monitor2;
  if (HAL_RAMECC_Init(&hramecc1_m2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC1 M3 : D0TCM-RAM
  */
  hramecc1_m3.Instance = RAMECC1_Monitor3;
  if (HAL_RAMECC_Init(&hramecc1_m3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC1 M4 : D1TCM-RAM
  */
  hramecc1_m4.Instance = RAMECC1_Monitor4;
  if (HAL_RAMECC_Init(&hramecc1_m4) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC1 M5 : ETM RAM
  */
  hramecc1_m5.Instance = RAMECC1_Monitor5;
  if (HAL_RAMECC_Init(&hramecc1_m5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC2 M1 : SRAM1_0
  */
  hramecc2_m1.Instance = RAMECC2_Monitor1;
  if (HAL_RAMECC_Init(&hramecc2_m1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC2 M2 SRAM1_1
  */
  hramecc2_m2.Instance = RAMECC2_Monitor2;
  if (HAL_RAMECC_Init(&hramecc2_m2) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC2 M3 : SRAM2_0
  */
  hramecc2_m3.Instance = RAMECC2_Monitor3;
  if (HAL_RAMECC_Init(&hramecc2_m3) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC2 M4 : SRAM2_1
  */
  hramecc2_m4.Instance = RAMECC2_Monitor4;
  if (HAL_RAMECC_Init(&hramecc2_m4) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC2 M5 : SRAM3
  */
  hramecc2_m5.Instance = RAMECC2_Monitor5;
  if (HAL_RAMECC_Init(&hramecc2_m5) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC3 M1 : SRAM4
  */
  hramecc3_m1.Instance = RAMECC3_Monitor1;
  if (HAL_RAMECC_Init(&hramecc3_m1) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initialize RAMECC3 M2 : Backup RAM
  */
  hramecc3_m2.Instance = RAMECC3_Monitor2;
  if (HAL_RAMECC_Init(&hramecc3_m2) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN RAMECC_Init 2 */

  /* USER CODE END RAMECC_Init 2 */

}

/**
  * @brief TIM1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_TIM1_Init(void)
{

  /* USER CODE BEGIN TIM1_Init 0 */

  /* USER CODE END TIM1_Init 0 */

  TIM_ClockConfigTypeDef sClockSourceConfig = {0};
  TIM_MasterConfigTypeDef sMasterConfig = {0};

  /* USER CODE BEGIN TIM1_Init 1 */

  /* USER CODE END TIM1_Init 1 */
  htim1.Instance = TIM1;
  htim1.Init.Prescaler = 240-1;
  htim1.Init.CounterMode = TIM_COUNTERMODE_UP;
  htim1.Init.Period = 65535;
  htim1.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
  htim1.Init.RepetitionCounter = 0;
  htim1.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_DISABLE;
  if (HAL_TIM_Base_Init(&htim1) != HAL_OK)
  {
    Error_Handler();
  }
  sClockSourceConfig.ClockSource = TIM_CLOCKSOURCE_INTERNAL;
  if (HAL_TIM_ConfigClockSource(&htim1, &sClockSourceConfig) != HAL_OK)
  {
    Error_Handler();
  }
  sMasterConfig.MasterOutputTrigger = TIM_TRGO_RESET;
  sMasterConfig.MasterOutputTrigger2 = TIM_TRGO2_RESET;
  sMasterConfig.MasterSlaveMode = TIM_MASTERSLAVEMODE_DISABLE;
  if (HAL_TIMEx_MasterConfigSynchronization(&htim1, &sMasterConfig) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN TIM1_Init 2 */

  /* USER CODE END TIM1_Init 2 */

}

/**
  * @brief USART1 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART1_UART_Init(void)
{

  /* USER CODE BEGIN USART1_Init 0 */

  /* USER CODE END USART1_Init 0 */

  /* USER CODE BEGIN USART1_Init 1 */

  /* USER CODE END USART1_Init 1 */
  huart1.Instance = USART1;
  huart1.Init.BaudRate = 1000000;
  huart1.Init.WordLength = UART_WORDLENGTH_8B;
  huart1.Init.StopBits = UART_STOPBITS_1;
  huart1.Init.Parity = UART_PARITY_NONE;
  huart1.Init.Mode = UART_MODE_TX_RX;
  huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart1.Init.OverSampling = UART_OVERSAMPLING_16;
  huart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart1.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart1, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart1, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart1) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART1_Init 2 */

  /* USER CODE END USART1_Init 2 */

}

/**
  * @brief USART3 Initialization Function
  * @param None
  * @retval None
  */
void MX_USART3_UART_Init(void)
{

  /* USER CODE BEGIN USART3_Init 0 */

  /* USER CODE END USART3_Init 0 */

  /* USER CODE BEGIN USART3_Init 1 */

  /* USER CODE END USART3_Init 1 */
  huart3.Instance = USART3;
  huart3.Init.BaudRate = 115200;
  huart3.Init.WordLength = UART_WORDLENGTH_8B;
  huart3.Init.StopBits = UART_STOPBITS_1;
  huart3.Init.Parity = UART_PARITY_NONE;
  huart3.Init.Mode = UART_MODE_TX_RX;
  huart3.Init.HwFlowCtl = UART_HWCONTROL_NONE;
  huart3.Init.OverSampling = UART_OVERSAMPLING_16;
  huart3.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
  huart3.Init.ClockPrescaler = UART_PRESCALER_DIV1;
  huart3.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;
  if (HAL_UART_Init(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetTxFifoThreshold(&huart3, UART_TXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_SetRxFifoThreshold(&huart3, UART_RXFIFO_THRESHOLD_1_8) != HAL_OK)
  {
    Error_Handler();
  }
  if (HAL_UARTEx_DisableFifoMode(&huart3) != HAL_OK)
  {
    Error_Handler();
  }
  /* USER CODE BEGIN USART3_Init 2 */

  /* USER CODE END USART3_Init 2 */

}

/**
  * Enable DMA controller clock
  */
static void MX_DMA_Init(void)
{

  /* DMA controller clock enable */
  __HAL_RCC_DMA1_CLK_ENABLE();

  /* DMA interrupt init */
  /* DMA1_Stream1_IRQn interrupt configuration */
  HAL_NVIC_SetPriority(DMA1_Stream1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(DMA1_Stream1_IRQn);

}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};
/* USER CODE BEGIN MX_GPIO_Init_1 */
/* USER CODE END MX_GPIO_Init_1 */

  /* GPIO Ports Clock Enable */
  __HAL_RCC_GPIOE_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();
  __HAL_RCC_GPIOD_CLK_ENABLE();
  __HAL_RCC_GPIOC_CLK_ENABLE();

  /*Configure GPIO pin Output Level */
  HAL_GPIO_WritePin(SPI4_CS_GPIO_Port, SPI4_CS_Pin, GPIO_PIN_SET);

  /*Configure GPIO pin : SPI4_CS_Pin */
  GPIO_InitStruct.Pin = SPI4_CS_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_VERY_HIGH;
  HAL_GPIO_Init(SPI4_CS_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : Start_recording_Pin */
  GPIO_InitStruct.Pin = Start_recording_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(Start_recording_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pin : Load_SD_file_to_network_Pin */
  GPIO_InitStruct.Pin = Load_SD_file_to_network_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_RISING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(Load_SD_file_to_network_GPIO_Port, &GPIO_InitStruct);

  /*Configure GPIO pins : Send_recording_Pin Generate_AI_Pin */
  GPIO_InitStruct.Pin = Send_recording_Pin|Generate_AI_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_PULLUP;
  HAL_GPIO_Init(GPIOD, &GPIO_InitStruct);

  /* EXTI interrupt init*/
  HAL_NVIC_SetPriority(EXTI0_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI0_IRQn);

  HAL_NVIC_SetPriority(EXTI1_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI1_IRQn);

  HAL_NVIC_SetPriority(EXTI3_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI3_IRQn);

  HAL_NVIC_SetPriority(EXTI15_10_IRQn, 0, 0);
  HAL_NVIC_EnableIRQ(EXTI15_10_IRQn);

/* USER CODE BEGIN MX_GPIO_Init_2 */
/* USER CODE END MX_GPIO_Init_2 */
}

/* USER CODE BEGIN 4 */

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance == USART3) // sprawdź który UART
	{

	}
}

void HAL_UART_TxCpltCallback(UART_HandleTypeDef *huart)
{
	if (huart->Instance == USART3)
	{
		// Transmisja zakończona — możesz np. ustawić flagę lub wysłać kolejne dane

	}
}

void HAL_GPIO_EXTI_Callback(uint16_t GPIO_Pin)
{
	switch(GPIO_Pin)
	{
	case Start_recording_Pin:
		app_states = State_recording;
		break;
	case Send_recording_Pin:
		app_states = State_send;
		break;
	case Generate_AI_Pin:
		app_states = State_AI_result;
		break;
	default:
		break;
	}
}


/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
	/* User can add his own implementation to report the HAL error return state */
	__disable_irq();
	while (1)
	{
	}
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
	/* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
