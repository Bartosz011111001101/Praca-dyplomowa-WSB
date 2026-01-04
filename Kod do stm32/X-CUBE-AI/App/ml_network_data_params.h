/**
  ******************************************************************************
  * @file    ml_network_data_params.h
  * @author  AST Embedded Analytics Research Platform
  * @date    2026-01-04T13:33:27+0100
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * Copyright (c) 2026 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  ******************************************************************************
  */

#ifndef ML_NETWORK_DATA_PARAMS_H
#define ML_NETWORK_DATA_PARAMS_H

#include "ai_platform.h"

/*
#define AI_ML_NETWORK_DATA_WEIGHTS_PARAMS \
  (AI_HANDLE_PTR(&ai_ml_network_data_weights_params[1]))
*/

#define AI_ML_NETWORK_DATA_CONFIG               (NULL)


#define AI_ML_NETWORK_DATA_ACTIVATIONS_SIZES \
  { 1536, }
#define AI_ML_NETWORK_DATA_ACTIVATIONS_SIZE     (1536)
#define AI_ML_NETWORK_DATA_ACTIVATIONS_COUNT    (1)
#define AI_ML_NETWORK_DATA_ACTIVATION_1_SIZE    (1536)



#define AI_ML_NETWORK_DATA_WEIGHTS_SIZES \
  { 250152, }
#define AI_ML_NETWORK_DATA_WEIGHTS_SIZE         (250152)
#define AI_ML_NETWORK_DATA_WEIGHTS_COUNT        (1)
#define AI_ML_NETWORK_DATA_WEIGHT_1_SIZE        (250152)



#define AI_ML_NETWORK_DATA_ACTIVATIONS_TABLE_GET() \
  (&g_ml_network_activations_table[1])

extern ai_handle g_ml_network_activations_table[1 + 2];



#define AI_ML_NETWORK_DATA_WEIGHTS_TABLE_GET() \
  (&g_ml_network_weights_table[1])

extern ai_handle g_ml_network_weights_table[1 + 2];


#endif    /* ML_NETWORK_DATA_PARAMS_H */
