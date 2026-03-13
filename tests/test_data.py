import numpy as np
import os
import tempfile
import pytest

from data.generator import generate_time_series
from data.preprocessing import create_rolling_windows
from data.monash_loader import simulate_monash_dataset, generate_zero_shot_eval_dataset, parse_tsf

def test_generate_time_series():
    num_series = 5
    min_length = 50
    max_length = 100
    series_list = generate_time_series(num_series, min_length, max_length)

    assert len(series_list) == num_series
    for series in series_list:
        assert isinstance(series, np.ndarray)
        assert series.ndim == 1
        assert min_length <= len(series) <= max_length

def test_create_rolling_windows_normal():
    # Sequence of length 10
    series = np.arange(10, dtype=float)
    input_window = 3
    forecast_horizon = 2

    X, y, means, stds = create_rolling_windows(series, input_window, forecast_horizon)

    # Expected windows: 10 - 3 - 2 + 1 = 6 windows
    assert len(X) == 6
    assert len(y) == 6
    assert X.shape == (6, 3)
    assert y.shape == (6, 2)

    # Check the first window values: series[0:3] -> [0, 1, 2]
    expected_mean = np.mean([0, 1, 2])
    expected_std = np.std([0, 1, 2])

    assert np.isclose(means[0], expected_mean)
    assert np.isclose(stds[0], expected_std)
    assert np.allclose(X[0], (np.array([0, 1, 2]) - expected_mean) / expected_std)

    # Check that y is normalized with X's stats
    assert np.allclose(y[0], (np.array([3, 4]) - expected_mean) / expected_std)

def test_create_rolling_windows_too_short():
    series = np.arange(4, dtype=float)
    input_window = 3
    forecast_horizon = 2

    X, y, means, stds = create_rolling_windows(series, input_window, forecast_horizon)
    assert len(X) == 0
    assert len(y) == 0

def test_create_rolling_windows_zero_std():
    # Flat series should have standard deviation of 0, testing divide-by-zero avoidance
    series = np.ones(10, dtype=float)
    input_window = 3
    forecast_horizon = 2

    X, y, means, stds = create_rolling_windows(series, input_window, forecast_horizon)
    assert np.all(stds == 1.0) # Our logic sets stds to 1.0 when 0
    assert np.all(X == 0.0) # (1 - 1) / 1.0 = 0.0

def test_simulate_monash_dataset():
    num_series = 3
    series_list = simulate_monash_dataset(num_series)
    assert len(series_list) == num_series
    for series in series_list:
        assert isinstance(series, np.ndarray)
        assert len(series) >= 500

def test_generate_zero_shot_eval_dataset():
    num_series = 4
    min_len, max_len = 100, 200
    series_list = generate_zero_shot_eval_dataset(num_series, min_len, max_len)
    assert len(series_list) == num_series
    for series in series_list:
        assert isinstance(series, np.ndarray)

def test_parse_tsf():
    # Create a dummy .tsf file
    content = """@attribute series_name string
@attribute start_timestamp date
@data
T1:2020-01-01 00-00-00:1.0,2.0,3.0,?,5.0
T2:2020-01-01 00-00-00:10.5,20.5
"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.tsf', mode='w', encoding='ISO-8859-1') as f:
        f.write(content)
        temp_path = f.name

    try:
        series_list = parse_tsf(temp_path)
        assert len(series_list) == 2

        # Test first series handles missing value '?' correctly
        s1 = series_list[0]
        assert len(s1) == 5
        assert np.isnan(s1[3])
        assert s1[4] == 5.0

        # Test second series parsed correctly
        s2 = series_list[1]
        assert np.allclose(s2, np.array([10.5, 20.5]))

        # Test max_series
        limited_series = parse_tsf(temp_path, max_series=1)
        assert len(limited_series) == 1
    finally:
        os.remove(temp_path)
