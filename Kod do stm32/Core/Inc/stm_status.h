#ifndef __STM32_STATUS_H
#define __STM32_STATUS_H



typedef enum states_types
{
	State_idle = 0,
	State_recording,
	State_send,
	State_AI_result,
	State_recording_and_send_chunks,
	State_wait_for_next_step
}App_State;




#endif /* __STM32_STATUS_H */
