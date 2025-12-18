#include "mic_ai_pi.h"

#include <math.h>

#include "mic_ai_util.h"

void mic_ai_pi_init(mic_ai_pi_t* pi, float kp, float ki, float dt, uint8_t has_limit, float limit) {
    if (!pi) {
        return;
    }
    pi->kp = kp;
    pi->ki = ki;
    pi->dt = dt;
    pi->limit = limit;
    pi->integrator = 0.0f;
    pi->has_limit = has_limit ? 1 : 0;
}

void mic_ai_pi_reset(mic_ai_pi_t* pi) {
    if (!pi) {
        return;
    }
    pi->integrator = 0.0f;
}

float mic_ai_pi_step(mic_ai_pi_t* pi, float error) {
    if (!pi) {
        return 0.0f;
    }

    const float u_unsat = pi->kp * error + pi->integrator;
    float u = u_unsat;

    if (pi->has_limit) {
        const float lim = fabsf(pi->limit);
        u = mic_ai_clampf(u_unsat, -lim, lim);
        // anti-windup: integrate only when not saturated (matches control/vector_foc.py semantics)
        if (fabsf(u) < lim) {
            pi->integrator += error * pi->ki * pi->dt;
        }
    } else {
        pi->integrator += error * pi->ki * pi->dt;
        u = u_unsat;  // output uses previous integrator value, same as Python implementation
    }

    return u;
}

