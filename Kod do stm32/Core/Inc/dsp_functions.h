/*
 * dsp_functions.h
 *
 *  Created on: Sep 16, 2025
 *      Author: Bartosz
 */

#ifndef INC_DSP_FUNCTIONS_H_
#define INC_DSP_FUNCTIONS_H_

#include "main.h"

void compute_mfcc_features(const uint16_t* audio_buf, float* feature_vector, int len_samples);
void StartRecording(void);
void StopRecordingAndSend(void);
void split_and_send(uint32_t *data, int size) ;
void delay_us(uint16_t us);
void run_inference(volatile uint16_t *audio_buf, int num_samples);
void debug_msg(const char *msg);
void init_network(void);

#endif /* INC_DSP_FUNCTIONS_H_ */
