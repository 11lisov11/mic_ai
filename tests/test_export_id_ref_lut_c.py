import json
import re
import sys

from mic_ai.tools import export_id_ref_lut_c


def test_export_id_ref_lut_c(tmp_path, monkeypatch) -> None:
    data = {
        "omega_ref_grid": [1.0, 2.0],
        "load_grid": [0.1, 0.2],
        "lut": {
            "1|0.1": 0.5,
            "1|0.2": 0.6,
            "2|0.1": 0.7,
            "2|0.2": 0.8,
        },
    }
    lut_path = tmp_path / "id_ref_lut.json"
    out_path = tmp_path / "lut.h"
    lut_path.write_text(json.dumps(data), encoding="utf-8")

    argv = [
        "export_id_ref_lut_c",
        "--lut",
        str(lut_path),
        "--out",
        str(out_path),
        "--symbol-prefix",
        "test",
    ]
    monkeypatch.setattr(sys, "argv", argv)
    export_id_ref_lut_c.main()

    text = out_path.read_text(encoding="utf-8")
    assert "#define TEST_OMEGA_SIZE 2" in text
    assert "#define TEST_LOAD_SIZE 2" in text

    match = re.search(
        r"static const int16_t test_id_ref_table_q\[4\] = \{([^}]*)\};",
        text,
        re.S,
    )
    assert match is not None
    values = [int(x) for x in re.findall(r"-?\d+", match.group(1))]
    assert values == [512, 614, 717, 819]
