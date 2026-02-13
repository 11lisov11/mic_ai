from pathlib import Path

import torch

from mic_ai.ai.distill_voltage import build_tiny_student, export_tiny_to_c, export_tiny_to_c_header, load_teacher_policy
from mic_ai.ai.agents.ppo_voltage import ActorCritic


def test_distill_exports(tmp_path: Path) -> None:
    student = build_tiny_student(6, 2)
    json_path = export_tiny_to_c(student, tmp_path / "student.json")
    header_path = export_tiny_to_c_header(student, tmp_path / "student.h")
    assert json_path.exists()
    assert header_path.exists()


def test_load_teacher_policy(tmp_path: Path) -> None:
    model = ActorCritic(state_dim=7, action_dim=1, hidden_sizes=(64, 64))
    ckpt = tmp_path / "teacher.pth"
    torch.save(model.state_dict(), ckpt)
    teacher = load_teacher_policy(str(ckpt), state_dim=7, action_dim=1, hidden_sizes=(64, 64))
    assert teacher is not None
