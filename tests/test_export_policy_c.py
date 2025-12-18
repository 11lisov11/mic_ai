import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from mic_ai.ai.agents.ppo_voltage import ActorCritic
from mic_ai.tools.export_policy_c import export_actor_to_c_header


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


@pytest.mark.skipif(not _have_gcc(), reason="gcc not available")
def test_exported_actor_header_runs_in_c(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = ActorCritic(state_dim=8, action_dim=2, hidden_sizes=(64, 64))
    state = model.state_dict()

    ckpt = tmp_path / "actor.pth"
    torch.save(state, ckpt)

    actor_h = tmp_path / "actor.h"
    export_actor_to_c_header(ckpt, actor_h, symbol_prefix="mic_ai_actor")

    rng = np.random.default_rng(123)
    n_vec = 20
    obs = rng.normal(0.0, 1.0, size=(n_vec, 8)).astype(np.float32)

    with torch.no_grad():
        mu, _std, _value = model(torch.as_tensor(obs))
    exp = mu.detach().cpu().numpy().astype(np.float32)

    vectors_h = tmp_path / "vectors.h"
    vectors_h.write_text(
        "\n".join(
            [
                "#pragma once",
                f"#define N_VEC {n_vec}",
                "static const float obs[N_VEC][8] = {",
                *[f"  {{{_as_c_float_list(obs[i])}}}," for i in range(n_vec)],
                "};",
                "static const float exp_mu[N_VEC][2] = {",
                *[f"  {{{_as_c_float_list(exp[i])}}}," for i in range(n_vec)],
                "};",
                "",
            ]
        ),
        encoding="utf-8",
    )

    test_c = tmp_path / "test_actor.c"
    test_c.write_text(
        r"""
#include <math.h>
#include <stdio.h>

#include "mic_ai_nn.h"

#include "actor.h"
#include "vectors.h"

static void actor_forward(const float* x, float* y) {
    static float h1[MIC_AI_ACTOR_H1_DIM];
    static float h2[MIC_AI_ACTOR_H2_DIM];

    mic_ai_linear((const float*)mic_ai_actor_W0, mic_ai_actor_b0, x, h1, MIC_AI_ACTOR_H1_DIM, MIC_AI_ACTOR_IN_DIM);
    mic_ai_tanh_inplace(h1, MIC_AI_ACTOR_H1_DIM);
    mic_ai_linear((const float*)mic_ai_actor_W1, mic_ai_actor_b1, h1, h2, MIC_AI_ACTOR_H2_DIM, MIC_AI_ACTOR_H1_DIM);
    mic_ai_tanh_inplace(h2, MIC_AI_ACTOR_H2_DIM);
    mic_ai_linear((const float*)mic_ai_actor_W2, mic_ai_actor_b2, h2, y, MIC_AI_ACTOR_OUT_DIM, MIC_AI_ACTOR_H2_DIM);
    mic_ai_tanh_inplace(y, MIC_AI_ACTOR_OUT_DIM);
}

static int nearly_equal(float a, float b, float tol) { return fabsf(a - b) <= tol; }

int main(void) {
    const float tol = 5e-4f;
    for (int i = 0; i < N_VEC; i++) {
        float y[2] = {0};
        actor_forward(obs[i], y);
        for (int j = 0; j < 2; j++) {
            if (!nearly_equal(y[j], exp_mu[i][j], tol)) {
                printf("Mismatch vec %d[%d]: %.7f exp %.7f\n", i, j, y[j], exp_mu[i][j]);
                return 1;
            }
        }
    }
    return 0;
}
""".strip()
        + "\n",
        encoding="utf-8",
    )

    embedded_root = Path(__file__).resolve().parents[1] / "embedded"
    include_dir = embedded_root / "include"
    src_dir = embedded_root / "src"

    exe = tmp_path / "test_actor.exe"
    cmd = [
        "gcc",
        "-std=c99",
        "-O2",
        "-I",
        str(include_dir),
        "-I",
        str(tmp_path),
        str(src_dir / "mic_ai_nn.c"),
        str(test_c),
        "-lm",
        "-o",
        str(exe),
    ]
    build = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert build.returncode == 0, f"gcc failed:\n{build.stdout}\n{build.stderr}"

    run = subprocess.run([str(exe)], capture_output=True, text=True, check=False)
    assert run.returncode == 0, f"actor inference failed:\n{run.stdout}\n{run.stderr}"
