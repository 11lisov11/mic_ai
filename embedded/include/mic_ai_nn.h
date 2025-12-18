#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void mic_ai_linear(const float* W, const float* b, const float* x, float* y, int out_dim, int in_dim);
void mic_ai_tanh_inplace(float* x, int n);

#ifdef __cplusplus
}  // extern "C"
#endif

