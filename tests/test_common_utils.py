from __future__ import annotations

from pathlib import Path

from tools.common_utils import (
    json_dump,
    json_load,
    parse_csv_list,
    parse_int_list,
    read_csv,
    write_csv,
)


def test_parse_helpers() -> None:
    assert parse_csv_list(" air56, al31 ,,ao2 ") == ["air56", "al31", "ao2"]
    assert parse_int_list("101, 202,303") == [101, 202, 303]


def test_csv_json_roundtrip(tmp_path: Path) -> None:
    rows = [
        {"motor": "air56", "value": 1.0},
        {"motor": "al31", "value": 2.0},
    ]
    csv_path = tmp_path / "rows.csv"
    write_csv(csv_path, rows)
    got_rows = read_csv(csv_path)
    assert len(got_rows) == 2
    assert got_rows[0]["motor"] == "air56"
    assert got_rows[1]["motor"] == "al31"

    payload = {"ok": True, "count": 2}
    json_path = tmp_path / "payload.json"
    json_dump(json_path, payload)
    got_payload = json_load(json_path)
    assert got_payload == payload
