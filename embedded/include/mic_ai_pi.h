#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mic_ai_pi {
    float kp;
    float ki;
    float dt;
    float limit;
    float integrator;
    uint8_t has_limit;
} mic_ai_pi_t;

void mic_ai_pi_init(mic_ai_pi_t* pi, float kp, float ki, float dt, uint8_t has_limit, float limit);
void mic_ai_pi_reset(mic_ai_pi_t* pi);
float mic_ai_pi_step(mic_ai_pi_t* pi, float error);

#ifdef __cplusplus
}  // extern "C"
#endif

