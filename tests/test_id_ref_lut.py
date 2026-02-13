from control.id_ref_lut import IdRefLut


def test_id_ref_lut_exact_lookup() -> None:
    lut = IdRefLut(
        omega_grid=[0.0, 10.0],
        load_grid=[0.0, 5.0],
        lut={
            "0|0": 0.1,
            "0|5": 0.2,
            "10|0": 0.3,
            "10|5": 0.4,
        },
    )
    assert lut.query(0.0, 0.0) == 0.1
    assert lut.query(0.0, 5.0) == 0.2
    assert lut.query(10.0, 0.0) == 0.3
    assert lut.query(10.0, 5.0) == 0.4


def test_id_ref_lut_bilinear_interp() -> None:
    lut = IdRefLut(
        omega_grid=[0.0, 10.0],
        load_grid=[0.0, 10.0],
        lut={
            "0|0": 0.0,
            "0|10": 20.0,
            "10|0": 10.0,
            "10|10": 30.0,
        },
    )
    value = lut.query(5.0, 5.0)
    assert abs(value - 15.0) < 1e-6


def test_id_ref_lut_out_of_bounds_clamps() -> None:
    lut = IdRefLut(
        omega_grid=[0.0, 10.0],
        load_grid=[0.0, 10.0],
        lut={
            "0|0": 1.0,
            "0|10": 2.0,
            "10|0": 3.0,
            "10|10": 4.0,
        },
    )
    assert lut.query(-5.0, -2.0) == 1.0
    assert lut.query(50.0, 50.0) == 4.0
