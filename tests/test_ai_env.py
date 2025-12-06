import sys
from pathlib import Path

import numpy as np

# Ensure project root on path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mic_ai.ai.ai_env import AiEnvConfig, MicAiAIEnv
from mic_ai.ai.curiosity import SimpleCuriosityModule
from mic_ai.ai.simple_agent import SimpleAdaptiveAgent
from mic_ai.core.env import make_env_from_config
from simulation.gym_env import InductionMotorEnv


def test_ai_env_runs_one_episode():
    cfg_path = Path("config/env_demo_true_motor1.py")
    env_base = make_env_from_config(str(cfg_path))
    env_cfg = env_base.env_config
    ai_cfg = AiEnvConfig(
        episode_steps=15,
        dt=float(env_cfg.sim.dt),
        omega_ref=float(2.0 * np.pi * 20.0 / max(env_cfg.motor.p, 1)),
        w_speed_error=0.0,
        w_current_rms=0.0,
        control_mode="foc_assist",
        i_base=1.0,
    )
    training_env = InductionMotorEnv(env_cfg)
    ai_env = MicAiAIEnv(training_env, ai_cfg, curiosity=SimpleCuriosityModule(weight=0.0))
    agent = SimpleAdaptiveAgent(
        feature_keys=["omega_norm", "omega_ref_norm", "err_norm", "id_norm", "iq_norm", "prev_delta_norm"],
        action_scale=1.0,
        lr=1e-3,
    )

    obs = ai_env.reset()
    done = False
    steps = 0
    while not done and steps < ai_cfg.episode_steps + 5:
        delta_rel = agent.act(obs)
        obs, reward, done, info = ai_env.step(delta_rel)
        agent.record_reward(reward)
        assert isinstance(reward, float)
        assert "speed_error" in info
        steps += 1

    agent.update_after_episode()
    assert steps > 0
