from bench.orchestrator import ExperimentConfig, run_experiment
from sim.safety import SafetyLimits, SafetySupervisor


def test_safety_inst_current_limit() -> None:
    limits = SafetyLimits(i_inst_max=5.0)
    supervisor = SafetySupervisor(limits)

    ok, reason = supervisor.check(
        state={"omega": 0.0},
        signals={"i_abc": (6.0, 0.0, 0.0)},
    )

    assert not ok
    assert "i_inst_max" in reason


def test_orchestrator_terminates_on_omega() -> None:
    limits = SafetyLimits(omega_max=50.0)
    supervisor = SafetySupervisor(limits)

    def signal_fn(t: float, step: int) -> tuple[dict[str, float], dict[str, object]]:
        omega = 0.0 if t < 0.05 else 100.0
        return {"omega": omega}, {"i_abc": (0.0, 0.0, 0.0)}

    config = ExperimentConfig(
        dt=0.01,
        duration=0.2,
        safety=supervisor,
        signal_fn=signal_fn,
    )
    result = run_experiment(config, policy=None)

    assert not result.ok
    assert result.terminated_early
    assert result.safety_trip
    assert "omega_max" in result.reason
