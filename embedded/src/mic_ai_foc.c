#include "mic_ai_foc.h"

#include <math.h>

#include "mic_ai_transform.h"
#include "mic_ai_util.h"

void mic_ai_foc_init(mic_ai_foc_controller_t* foc,
                     const mic_ai_foc_params_t* params,
                     int pole_pairs,
                     float Rr,
                     float Lr,
                     float dt) {
    if (!foc || !params) {
        return;
    }

    foc->params = *params;
    foc->pole_pairs = pole_pairs;
    foc->Rr = Rr;
    foc->Lr = Lr;
    foc->dt = dt;

    mic_ai_pi_init(&foc->pi_id, params->kp_id, params->ki_id, dt, params->has_v_limit, params->v_limit);
    mic_ai_pi_init(&foc->pi_iq, params->kp_iq, params->ki_iq, dt, params->has_v_limit, params->v_limit);
    mic_ai_pi_init(&foc->pi_speed, params->kp_speed, params->ki_speed, dt, params->has_iq_limit, params->iq_limit);

    foc->theta_e = 0.0f;
    foc->omega_syn = 0.0f;
    foc->last_iq_ref = 0.0f;
    foc->last_id_ref = 0.0f;
    foc->max_di_dt = 500.0f;  // A/s, matches Python default
}

void mic_ai_foc_reset(mic_ai_foc_controller_t* foc) {
    if (!foc) {
        return;
    }
    mic_ai_pi_reset(&foc->pi_id);
    mic_ai_pi_reset(&foc->pi_iq);
    mic_ai_pi_reset(&foc->pi_speed);
    foc->theta_e = 0.0f;
    foc->omega_syn = 0.0f;
    foc->last_iq_ref = 0.0f;
    foc->last_id_ref = 0.0f;
}

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
                     float* i_q_ref_out) {
    (void)t;
    (void)torque_e;
    (void)theta_mech;

    if (!foc) {
        if (v_d) {
            *v_d = 0.0f;
        }
        if (v_q) {
            *v_q = 0.0f;
        }
        if (theta_e_used) {
            *theta_e_used = 0.0f;
        }
        if (omega_syn) {
            *omega_syn = 0.0f;
        }
        if (i_d_ref_out) {
            *i_d_ref_out = 0.0f;
        }
        if (i_q_ref_out) {
            *i_q_ref_out = 0.0f;
        }
        return;
    }

    // Use current internal electrical angle for Park transform (matches Python code)
    const float theta_e = foc->theta_e;
    float i_d = 0.0f;
    float i_q_local = 0.0f;
    mic_ai_abc_to_dq(i_a, i_b, i_c, theta_e, &i_d, &i_q_local);

    const float e_speed = omega_ref - omega_m;
    float i_q_ref = mic_ai_pi_step(&foc->pi_speed, e_speed);
    if (foc->params.has_iq_limit) {
        const float lim = fabsf(foc->params.iq_limit);
        i_q_ref = mic_ai_clampf(i_q_ref, -lim, lim);
    }
    float i_d_ref = foc->params.id_ref;

    // Slew-rate limit current references (matches Python: max_di_dt * dt)
    const float max_delta = foc->max_di_dt * foc->dt;
    i_q_ref = mic_ai_clampf(i_q_ref, foc->last_iq_ref - max_delta, foc->last_iq_ref + max_delta);
    i_d_ref = mic_ai_clampf(i_d_ref, foc->last_id_ref - max_delta, foc->last_id_ref + max_delta);
    foc->last_iq_ref = i_q_ref;
    foc->last_id_ref = i_d_ref;

    const float e_id = i_d_ref - i_d;
    const float e_iq = i_q_ref - i_q_local;

    float v_d_local = mic_ai_pi_step(&foc->pi_id, e_id);
    float v_q_local = mic_ai_pi_step(&foc->pi_iq, e_iq);

    // Voltage magnitude limit on dq vector.
    if (foc->params.has_v_limit) {
        const float lim = fabsf(foc->params.v_limit);
        const float mag = mic_ai_hypotf(v_d_local, v_q_local);
        if (mag > lim && mag > 0.0f) {
            const float scale = lim / mag;
            v_d_local *= scale;
            v_q_local *= scale;
        }
    }

    // Slip estimate and synchronous speed update.
    const float eps = 1e-6f;
    float omega_slip = 0.0f;
    if (foc->Rr > 0.0f && foc->Lr > eps) {
        const float denom = fmaxf(fabsf(i_d_ref), eps);
        omega_slip = (foc->Rr / foc->Lr) * (i_q_ref / denom);
    }
    const float omega_syn_local = (float)foc->pole_pairs * omega_m + omega_slip;
    foc->omega_syn = omega_syn_local;
    foc->theta_e = theta_e + omega_syn_local * foc->dt;

    if (v_d) {
        *v_d = v_d_local;
    }
    if (v_q) {
        *v_q = v_q_local;
    }
    if (theta_e_used) {
        *theta_e_used = theta_e;
    }
    if (omega_syn) {
        *omega_syn = omega_syn_local;
    }
    if (i_d_ref_out) {
        *i_d_ref_out = i_d_ref;
    }
    if (i_q_ref_out) {
        *i_q_ref_out = i_q_ref;
    }
}
