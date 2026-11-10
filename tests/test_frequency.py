from ezero.frequency import bucket_name, pool_window, season_length


def test_lsf_frequencies():
    assert bucket_name("H") == "hour" and season_length("H") == 24
    assert bucket_name("15T") == "15min" and season_length("15T") == 96
    assert bucket_name("10T") == "10min" and season_length("10T") == 144
    assert bucket_name("5min") == "minute"
    assert bucket_name("W-SUN") == "week"
    assert bucket_name("MS") == "month"
    assert bucket_name("Q-DEC") == "quarter"
    assert bucket_name(None) == "unknown"


def test_legacy_patch_map_is_kept():
    """DTP windows keep the original EZERO frequency_patch_map."""
    assert pool_window("S") == 64
    assert pool_window("5T") == 32
    assert pool_window("H") == 16
    assert pool_window("D") == 8
    assert pool_window("W") == 4
    assert pool_window("M") == 2
