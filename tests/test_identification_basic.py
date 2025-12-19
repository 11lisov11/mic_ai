from bench import orchestrator
from bench.identification import IdentificationConfig, run_id_sequence
from config.env import create_default_env


def test_identification_basic(tmp_path) -> None:
    env = create_default_env()
    rs_nom = env.motor.Rs
    id_cfg = IdentificationConfig(
        env=env,
        dt=0.002,
        duration=1.0,
        pulse_start=0.1,
        pulse_end=0.8,
        pulse_v_q=0.5 * env.motor.I_n * rs_nom,
        rs_nom=rs_nom,
        rs_max=rs_nom * 2.0,
        log_root=str(tmp_path),
        run_name="id_basic",
        seed=1,
    )
    result = run_id_sequence(orchestrator, id_cfg)
    assert 0.0 <= result["aging"] <= 1.0
    assert result["est_params"]["Rs_est"] > 0.0
