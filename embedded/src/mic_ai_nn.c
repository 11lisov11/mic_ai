#include "mic_ai_nn.h"

#include <math.h>

void mic_ai_linear(const float* W, const float* b, const float* x, float* y, int out_dim, int in_dim) {
    for (int row = 0; row < out_dim; row++) {
        float acc = b ? b[row] : 0.0f;
        const float* w_row = W + (row * in_dim);
        for (int col = 0; col < in_dim; col++) {
            acc += w_row[col] * x[col];
        }
        y[row] = acc;
    }
}

void mic_ai_tanh_inplace(float* x, int n) {
    for (int i = 0; i < n; i++) {
        x[i] = tanhf(x[i]);
    }
}

