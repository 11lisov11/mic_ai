#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void mic_ai_abc_to_alpha_beta(float i_a, float i_b, float i_c, float* i_alpha, float* i_beta);
void mic_ai_alpha_beta_to_dq(float i_alpha, float i_beta, float theta_e, float* i_d, float* i_q);
void mic_ai_dq_to_alpha_beta(float v_d, float v_q, float theta_e, float* v_alpha, float* v_beta);
void mic_ai_alpha_beta_to_abc(float v_alpha, float v_beta, float* v_a, float* v_b, float* v_c);
void mic_ai_abc_to_dq(float i_a, float i_b, float i_c, float theta_e, float* i_d, float* i_q);
void mic_ai_dq_to_abc(float v_d, float v_q, float theta_e, float* v_a, float* v_b, float* v_c);

#ifdef __cplusplus
}  // extern "C"
#endif

