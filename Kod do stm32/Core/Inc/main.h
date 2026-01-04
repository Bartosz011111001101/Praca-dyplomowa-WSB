/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.h
  * @brief          : Header for main.c file.
  *                   This file contains the common defines of the application.
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

/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __MAIN_H
#define __MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* Includes ------------------------------------------------------------------*/
#include "stm32h7xx_hal.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Exported types ------------------------------------------------------------*/
/* USER CODE BEGIN ET */

/* USER CODE END ET */

/* Exported constants --------------------------------------------------------*/
/* USER CODE BEGIN EC */

/* USER CODE END EC */

/* Exported macro ------------------------------------------------------------*/
/* USER CODE BEGIN EM */
#define SD_SPI_HANDLE hspi4
#define ADC_BITS    14
#define N_MFCC      NUM_MFCC_COEFFS
#define N_MELS      NUM_MEL_FILTERS
#define N_FFT       (FRAME_SIZE)
#define FRAME_SIZE 1024
#define HOP_LENGTH 512
#define NUM_MFCC_COEFFS 40
#define NUM_MEL_FILTERS 40
#define SAMPLE_RATE     44100
#define RECORD_SECONDS  5U
#define NUM_SAMPLES     (SAMPLE_RATE * RECORD_SECONDS)
#define Button_Start_recording (HAL_GPIO_ReadPin(Start_recording_GPIO_Port, Start_recording_Pin) == GPIO_PIN_RESET)
#define Button_sending_record_Pressed (HAL_GPIO_ReadPin(Send_recording_GPIO_Port, Send_recording_Pin) == GPIO_PIN_RESET)
#define Button_generate_AI (HAL_GPIO_ReadPin(Generate_AI_GPIO_Port, Generate_AI_Pin) == GPIO_PIN_RESET)
#define Load_SD_file_to_network (HAL_GPIO_ReadPin(Load_SD_file_to_network_Port, Load_SD_file_to_network_Pin) == GPIO_PIN_RESET)
#define NUM_FRAMES ((NUM_SAMPLES - FRAME_SIZE) / HOP_LENGTH + 1)

/* USER CODE END EM */

/* Exported functions prototypes ---------------------------------------------*/
void Error_Handler(void);
void MX_USART3_UART_Init(void);

/* USER CODE BEGIN EFP */
void zacznij_generowac();
/* USER CODE END EFP */

/* Private defines -----------------------------------------------------------*/
#define SPI4_CS_Pin GPIO_PIN_4
#define SPI4_CS_GPIO_Port GPIOE
#define Start_recording_Pin GPIO_PIN_3
#define Start_recording_GPIO_Port GPIOA
#define Start_recording_EXTI_IRQn EXTI3_IRQn
#define Load_SD_file_to_network_Pin GPIO_PIN_12
#define Load_SD_file_to_network_GPIO_Port GPIOC
#define Load_SD_file_to_network_EXTI_IRQn EXTI15_10_IRQn
#define Send_recording_Pin GPIO_PIN_0
#define Send_recording_GPIO_Port GPIOD
#define Send_recording_EXTI_IRQn EXTI0_IRQn
#define Generate_AI_Pin GPIO_PIN_1
#define Generate_AI_GPIO_Port GPIOD
#define Generate_AI_EXTI_IRQn EXTI1_IRQn

/* USER CODE BEGIN Private defines */
extern void compute_mfcc_features(const uint16_t* audio_buf, float* feature_vector, int len_samples);
extern void StartRecording(void);
extern void StopRecordingAndSend(void);
extern void split_and_send(uint32_t *data, int size) ;
extern void delay_us(uint16_t us);
//extern void run_inference(volatile uint16_t *audio_buf);
extern void run_inference(volatile uint16_t *audio_buf, int num_samples);
extern void debug_msg(const char *msg);
extern void init_network(void);
extern I2C_HandleTypeDef hi2c4; // Add this line



/* USER CODE END Private defines */

#ifdef __cplusplus
}
#endif

#endif /* __MAIN_H */
