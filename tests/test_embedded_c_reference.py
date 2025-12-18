import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from config.env import ENV, FocParams
from control.vector_foc import FocController


def _have_gcc() -> bool:
    return shutil.which("gcc") is not None


def _c_float(value: float) -> str:
    text = f"{float(value):.9g}"
    if "e" not in text.lower() and "." not in text:
        text += ".0"
    return f"{text}f"


def _as_c_float_list(values: np.ndarray) -> str:
    arr = np.asarray(values, dtype=np.float32).ravel()
    return ", ".join(_c_float(v) for v in arr.tolist())


def _quantize_foc_state(ctrl: FocController) -> None:
    def f32(x: float) -> float:
        return float(np.float32(x))

    ctrl.pi_id.integrator = f32(ctrl.pi_id.integrator)
    ctrl.pi_iq.integrator = f32(ctrl.pi_iq.integrator)
    ctrl.pi_speed.integrator = f32(ctrl.pi_speed.integrator)
    ctrl.theta_e = f32(ctrl.theta_e)
    ctrl.omega_syn = f32(ctrl.omega_syn)
    ctrl.last_iq_ref = f32(ctrl.last_iq_ref)
    ctrl.last_id_ref = f32(ctrl.last_id_ref)


@pytest.mark.skipif(not _have_gcc(), reason="gcc not available")
def test_embedded_c_foc_matches_python(tmp_path: Path) -> None:
    dt = 1e-4
    foc_params = FocParams(
        kp_id=1.2,
        ki_id=120.0,
        kp_iq=1.1,
        ki_iq=110.0,
        kp_speed=0.4,
        ki_speed=3.0,
        id_ref=0.4,
        iq_limit=2.0,
        v_limit=50.0,
    )
    motor = ENV.motor
    foc = FocController(foc_params, motor, dt=float(dt))
    foc.reset()

    pole_pairs = int(getattr(motor, "p", 2))
    rr = float(getattr(motor, "Rr", 0.0))
    lr = float(getattr(motor, "Lr_sigma", 0.0) + getattr(motor, "Lm", 0.0))

    rng = np.random.default_rng(0)
    n_steps = 200

    omega_ref = rng.uniform(-120.0, 120.0, size=n_steps).astype(np.float32)
    omega_m = rng.uniform(-80.0, 80.0, size=n_steps).astype(np.float32)
    i_abc = rng.uniform(-3.0, 3.0, size=(n_steps, 3)).astype(np.float32)

    exp_vd = np.zeros(n_steps, dtype=np.float32)
    exp_vq = np.zeros(n_steps, dtype=np.float32)
    exp_theta = np.zeros(n_steps, dtype=np.float32)
    exp_omega_syn = np.zeros(n_steps, dtype=np.float32)
    exp_id_ref = np.zeros(n_steps, dtype=np.float32)
    exp_iq_ref = np.zeros(n_steps, dtype=np.float32)

    for k in range(n_steps):
        v_d, v_q, theta_e, omega_syn, info = foc.step(
            t=float(k) * float(dt),
            omega_ref=float(omega_ref[k]),
            omega_m=float(omega_m[k]),
            i_abc=(float(i_abc[k, 0]), float(i_abc[k, 1]), float(i_abc[k, 2])),
            torque_e=0.0,
            theta_mech=0.0,
        )
        _quantize_foc_state(foc)
        exp_vd[k] = np.float32(v_d)
        exp_vq[k] = np.float32(v_q)
        exp_theta[k] = np.float32(theta_e)
        exp_omega_syn[k] = np.float32(omega_syn)
        exp_id_ref[k] = np.float32(info.get("i_d_ref", 0.0))
        exp_iq_ref[k] = np.float32(info.get("i_q_ref", 0.0))

    vectors_h = tmp_path / "vectors.h"
    vectors_h.write_text(
        "\n".join(
            [
                "#pragma once",
                f"#define N_STEPS {n_steps}",
                f"static const float omega_ref[N_STEPS] = {{{_as_c_float_list(omega_ref)}}};",
                f"static const float omega_m[N_STEPS] = {{{_as_c_float_list(omega_m)}}};",
                f"static const float i_a[N_STEPS] = {{{_as_c_float_list(i_abc[:, 0])}}};",
                f"static const float i_b[N_STEPS] = {{{_as_c_float_list(i_abc[:, 1])}}};",
                f"static const float i_c[N_STEPS] = {{{_as_c_float_list(i_abc[:, 2])}}};",
                f"static const float exp_vd[N_STEPS] = {{{_as_c_float_list(exp_vd)}}};",
                f"static const float exp_vq[N_STEPS] = {{{_as_c_float_list(exp_vq)}}};",
                f"static const float exp_theta[N_STEPS] = {{{_as_c_float_list(exp_theta)}}};",
                f"static const float exp_omega_syn[N_STEPS] = {{{_as_c_float_list(exp_omega_syn)}}};",
                f"static const float exp_id_ref[N_STEPS] = {{{_as_c_float_list(exp_id_ref)}}};",
                f"static const float exp_iq_ref[N_STEPS] = {{{_as_c_float_list(exp_iq_ref)}}};",
                "",
            ]
        ),
        encoding="utf-8",
    )

    test_c = tmp_path / "test_foc.c"
    c_src = r"""
#include <math.h>
#include <stdio.h>

#include "mic_ai_foc.h"
#include "vectors.h"

static int nearly_equal(float a, float b, float tol) { return fabsf(a - b) <= tol; }

int main(void) {
    mic_ai_foc_controller_t foc;
    mic_ai_foc_params_t params = {
        .kp_id = 1.2f,
        .ki_id = 120.0f,
        .kp_iq = 1.1f,
        .ki_iq = 110.0f,
        .kp_speed = 0.4f,
        .ki_speed = 3.0f,
        .id_ref = 0.4f,
        .has_iq_limit = 1,
        .iq_limit = 2.0f,
        .has_v_limit = 1,
        .v_limit = 50.0f,
    };

    const int pole_pairs = @POLE_PAIRS@;
    const float Rr = @RR@;
    const float Lr = @LR@;
    const float dt = 1e-4f;
    mic_ai_foc_init(&foc, &params, pole_pairs, Rr, Lr, dt);
    mic_ai_foc_reset(&foc);

    const float tol_v = 2e-3f;
    const float tol_w = 2e-3f;
    const float tol_theta = 2e-3f;
    const float tol_i = 2e-3f;

    for (int k = 0; k < N_STEPS; k++) {
        float v_d = 0.0f, v_q = 0.0f, theta_used = 0.0f, omega_syn = 0.0f, id_ref = 0.0f, iq_ref = 0.0f;
        mic_ai_foc_step(&foc,
                        (float)k * dt,
                        omega_ref[k],
                        omega_m[k],
                        i_a[k],
                        i_b[k],
                        i_c[k],
                        0.0f,
                        0.0f,
                        &v_d,
                        &v_q,
                        &theta_used,
                        &omega_syn,
                        &id_ref,
                        &iq_ref);

        if (!nearly_equal(v_d, exp_vd[k], tol_v) || !nearly_equal(v_q, exp_vq[k], tol_v)) {
            printf("Mismatch step %d: v=(%.6f, %.6f) exp=(%.6f, %.6f)\n", k, v_d, v_q, exp_vd[k], exp_vq[k]);
            return 1;
        }
        if (!nearly_equal(theta_used, exp_theta[k], tol_theta)) {
            printf("Mismatch step %d: theta=%.6f exp=%.6f\n", k, theta_used, exp_theta[k]);
            return 2;
        }
        if (!nearly_equal(omega_syn, exp_omega_syn[k], tol_w)) {
            printf("Mismatch step %d: omega_syn=%.6f exp=%.6f\n", k, omega_syn, exp_omega_syn[k]);
            return 3;
        }
        if (!nearly_equal(id_ref, exp_id_ref[k], tol_i) || !nearly_equal(iq_ref, exp_iq_ref[k], tol_i)) {
            printf("Mismatch step %d: i_ref=(%.6f, %.6f) exp=(%.6f, %.6f)\n", k, id_ref, iq_ref, exp_id_ref[k], exp_iq_ref[k]);
            return 4;
        }
    }
    return 0;
}
""".strip()
    c_src = c_src.replace("@POLE_PAIRS@", str(pole_pairs))
    c_src = c_src.replace("@RR@", f"{rr:.9g}f")
    c_src = c_src.replace("@LR@", f"{lr:.9g}f")
    test_c.write_text(c_src + "\n", encoding="utf-8")

    embedded_root = Path(__file__).resolve().parents[1] / "embedded"
    include_dir = embedded_root / "include"
    src_dir = embedded_root / "src"

    exe = tmp_path / "test_foc.exe"
    cmd = [
        "gcc",
        "-std=c99",
        "-O2",
        "-I",
        str(include_dir),
        str(src_dir / "mic_ai_transform.c"),
        str(src_dir / "mic_ai_pi.c"),
        str(src_dir / "mic_ai_foc.c"),
        str(test_c),
        "-lm",
        "-o",
        str(exe),
    ]
    build = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert build.returncode == 0, f"gcc failed:\n{build.stdout}\n{build.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, f"embedded foc test failed:\n{run.stdout}\n{run.stderr}"
