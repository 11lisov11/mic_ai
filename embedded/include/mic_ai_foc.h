#pragma once

#include <stdint.h>

#include "mic_ai_pi.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct mic_ai_foc_params {
    float kp_id;
    float ki_id;
    float kp_iq;
    float ki_iq;
    float kp_speed;
    float ki_speed;
    float id_ref;
    uint8_t has_iq_limit;
    float iq_limit;
    uint8_t has_v_limit;
    float v_limit;
} mic_ai_foc_params_t;

typedef struct mic_ai_foc_controller {
    mic_ai_foc_params_t params;
    int pole_pairs;
    float Rr;
    float Lr;
    float dt;

    mic_ai_pi_t pi_id;
    mic_ai_pi_t pi_iq;
    mic_ai_pi_t pi_speed;

    float theta_e;
    float omega_syn;
    float last_iq_ref;
    float last_id_ref;
    float max_di_dt;
} mic_ai_foc_controller_t;

void mic_ai_foc_init(mic_ai_foc_controller_t* foc,
                     const mic_ai_foc_params_t* params,
                     int pole_pairs,
                     float Rr,
                     float Lr,
                     float dt);

void mic_ai_foc_reset(mic_ai_foc_controller_t* foc);

void mic_ai_foc_step(mic_ai_foc_controller_t* foc,
                     float t,
                     float omega_ref,
                     float omega_m,
                     float i_a,
                     float i_b,
                     float i_c,
                     float torque_e,
                     float theta_mech,
                     float* v_d,
                     float* v_q,
                     float* theta_e_used,
                     float* omega_syn,
                     float* i_d_ref_out,
                     float* i_q_ref_out);

#ifdef __cplusplus
}  // extern "C"
#endif

