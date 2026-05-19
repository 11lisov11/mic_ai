from __future__ import annotations

from pathlib import Path

from tools.build_train3_refresh_release import build_refresh_manifest


def test_build_refresh_manifest_promotes_only_noncanonical_checkpoint(tmp_path: Path) -> None:
    air_ckpt = tmp_path / "air_canonical.pth"
    al_ckpt = tmp_path / "al31_actor_ep018.pth"
    promoted_al = tmp_path / "best_actor_step27_train3.pth"
    ao_ckpt = tmp_path / "ao2_canonical.pth"
    for path in (air_ckpt, al_ckpt, promoted_al, ao_ckpt):
        path.write_bytes(path.name.encode("ascii"))

    summary = tmp_path / "selected_recheck_summary.json"
    summary.write_text(
        """[
          {
            "motor": "air56",
            "acceptance_pass": true,
            "avg_power_saving_pct": 1.072,
            "avg_eta_gain_pct": 0.112,
            "start_stop_power_saving_pct": 2.140,
            "err_failures": 0.0,
            "envelope_fail_count": 0,
            "candidate_tag": "rand007_soft_track",
            "selected_is_canonical_baseline": true,
            "step27_selected_checkpoint": "%s"
          },
          {
            "motor": "al31",
            "acceptance_pass": true,
            "avg_power_saving_pct": 3.455,
            "avg_eta_gain_pct": 0.003,
            "start_stop_power_saving_pct": 13.736,
            "err_failures": 0.0,
            "envelope_fail_count": 0,
            "candidate_tag": "mid04_speed_dn_04",
            "selected_is_canonical_baseline": false,
            "step27_selected_checkpoint": "%s"
          },
          {
            "motor": "ao2",
            "acceptance_pass": true,
            "avg_power_saving_pct": 0.512,
            "avg_eta_gain_pct": 1.724,
            "start_stop_power_saving_pct": 0.0,
            "err_failures": 0.0,
            "envelope_fail_count": 0,
            "candidate_tag": "ao2_current_repro_rand017",
            "selected_is_canonical_baseline": true,
            "step27_selected_checkpoint": "%s"
          }
        ]"""
        % (air_ckpt.as_posix(), al_ckpt.as_posix(), ao_ckpt.as_posix()),
        encoding="utf-8",
    )
    joint_manifest = tmp_path / "joint.json"
    joint_manifest.write_text('{"protocol_hash": "joint"}', encoding="utf-8")
    fine_manifest = tmp_path / "fine.json"
    fine_manifest.write_text(
        """{
          "protocol_hash": "fine",
          "runs": [
            {
              "motor": "air56",
              "step27_promoted_checkpoint": "%s",
              "step27_included_canonical_checkpoint": "%s"
            },
            {
              "motor": "al31",
              "step27_promoted_checkpoint": "%s",
              "step27_included_canonical_checkpoint": "%s"
            },
            {
              "motor": "ao2",
              "step27_promoted_checkpoint": "%s",
              "step27_included_canonical_checkpoint": "%s"
            }
          ]
        }"""
        % (
            air_ckpt.as_posix(),
            air_ckpt.as_posix(),
            promoted_al.as_posix(),
            tmp_path.joinpath("old_al31.pth").as_posix(),
            ao_ckpt.as_posix(),
            ao_ckpt.as_posix(),
        ),
        encoding="utf-8",
    )

    manifest = build_refresh_manifest(
        summary_path=summary,
        joint_manifest_path=joint_manifest,
        finetune_manifest_path=fine_manifest,
    )

    assert manifest["research_refresh_complete"] is True
    rows = {row["motor"]: row for row in manifest["motors"]}
    assert rows["air56"]["decision"] == "keep_canonical_baseline"
    assert rows["al31"]["decision"] == "promote_training_checkpoint"
    assert rows["al31"]["release_checkpoint_exists"] is True
    assert rows["ao2"]["decision"] == "keep_canonical_baseline"
