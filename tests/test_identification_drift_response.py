from dataclasses import replace

from bench import orchestrator
from bench.identification import IdentificationConfig, run_id_sequence
from config.env import create_default_env


def test_identification_drift_response(tmp_path) -> None:
    env_nom = create_default_env()
    rs_nom = env_nom.motor.Rs
    env_drift = replace(env_nom, motor=replace(env_nom.motor, Rs=rs_nom * 1.5))

    pulse_v_q = 0.5 * env_nom.motor.I_n * rs_nom
    id_cfg_nom = IdentificationConfig(
        env=env_nom,
        dt=0.002,
        duration=1.0,
        pulse_start=0.1,
        pulse_end=0.8,
        pulse_v_q=pulse_v_q,
        rs_nom=rs_nom,
        rs_max=rs_nom * 2.0,
        log_root=str(tmp_path / "nom"),
        run_name="id_nom",
        seed=2,
    )
    id_cfg_drift = replace(
        id_cfg_nom,
        env=env_drift,
        log_root=str(tmp_path / "drift"),
        run_name="id_drift",
    )

    result_nom = run_id_sequence(orchestrator, id_cfg_nom)
    result_drift = run_id_sequence(orchestrator, id_cfg_drift)

    assert result_drift["aging"] > result_nom["aging"]
