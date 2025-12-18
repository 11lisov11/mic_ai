#include "mic_ai_transform.h"

#include <math.h>

static const float MIC_AI_SQRT3 = 1.7320508075688772f;
static const float MIC_AI_TWO_THIRDS = 0.6666666666666666f;  // 2/3

void mic_ai_abc_to_alpha_beta(float i_a, float i_b, float i_c, float* i_alpha, float* i_beta) {
    const float alpha = MIC_AI_TWO_THIRDS * (i_a - 0.5f * (i_b + i_c));
    const float beta = MIC_AI_TWO_THIRDS * (0.5f * MIC_AI_SQRT3) * (i_b - i_c);
    if (i_alpha) {
        *i_alpha = alpha;
    }
    if (i_beta) {
        *i_beta = beta;
    }
}

void mic_ai_alpha_beta_to_dq(float i_alpha, float i_beta, float theta_e, float* i_d, float* i_q) {
    const float cos_t = cosf(theta_e);
    const float sin_t = sinf(theta_e);
    const float d = i_alpha * cos_t + i_beta * sin_t;
    const float q = -i_alpha * sin_t + i_beta * cos_t;
    if (i_d) {
        *i_d = d;
    }
    if (i_q) {
        *i_q = q;
    }
}

void mic_ai_dq_to_alpha_beta(float v_d, float v_q, float theta_e, float* v_alpha, float* v_beta) {
    const float cos_t = cosf(theta_e);
    const float sin_t = sinf(theta_e);
    const float alpha = v_d * cos_t - v_q * sin_t;
    const float beta = v_d * sin_t + v_q * cos_t;
    if (v_alpha) {
        *v_alpha = alpha;
    }
    if (v_beta) {
        *v_beta = beta;
    }
}

void mic_ai_alpha_beta_to_abc(float v_alpha, float v_beta, float* v_a, float* v_b, float* v_c) {
    const float a = v_alpha;
    const float b = -0.5f * v_alpha + (MIC_AI_SQRT3 * 0.5f) * v_beta;
    const float c = -0.5f * v_alpha - (MIC_AI_SQRT3 * 0.5f) * v_beta;
    if (v_a) {
        *v_a = a;
    }
    if (v_b) {
        *v_b = b;
    }
    if (v_c) {
        *v_c = c;
    }
}

void mic_ai_abc_to_dq(float i_a, float i_b, float i_c, float theta_e, float* i_d, float* i_q) {
    float i_alpha = 0.0f;
    float i_beta = 0.0f;
    mic_ai_abc_to_alpha_beta(i_a, i_b, i_c, &i_alpha, &i_beta);
    mic_ai_alpha_beta_to_dq(i_alpha, i_beta, theta_e, i_d, i_q);
}

void mic_ai_dq_to_abc(float v_d, float v_q, float theta_e, float* v_a, float* v_b, float* v_c) {
    float v_alpha = 0.0f;
    float v_beta = 0.0f;
    mic_ai_dq_to_alpha_beta(v_d, v_q, theta_e, &v_alpha, &v_beta);
    mic_ai_alpha_beta_to_abc(v_alpha, v_beta, v_a, v_b, v_c);
}

