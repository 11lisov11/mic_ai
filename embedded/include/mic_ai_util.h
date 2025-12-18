#pragma once

#include <math.h>

static inline float mic_ai_clampf(float x, float lo, float hi) {
    return (x < lo) ? lo : (x > hi) ? hi : x;
}

static inline float mic_ai_hypotf(float x, float y) {
#if defined(__STDC_VERSION__) && (__STDC_VERSION__ >= 199901L)
    return hypotf(x, y);
#else
    return sqrtf(x * x + y * y);
#endif
}

