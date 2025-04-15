import math
import random
import numpy as np
import pytest

from chunk_manager.chunk_partitioner import split_wc_into_chunks, _allocate_extras


@pytest.fixture(autouse=True)
def setup_random_seed():
    """Set random seeds for reproducibility."""
    random.seed(42)
    np.random.seed(42)


def test_valid_inputs_basic_functionality():
    """Test basic valid input scenarios."""
    # Standard case
    total_wc = 1000
    min_wc = 50
    max_wc = 150
    result = split_wc_into_chunks(total_wc, min_wc, max_wc)
    assert isinstance(result, list)
    assert len(result) > 0
    assert sum(result) == total_wc
    for wc in result:
        assert min_wc <= wc <= max_wc
    
    # Min equals max case
    total_wc = 100
    min_wc = max_wc = 10
    result = split_wc_into_chunks(total_wc, min_wc, max_wc)
    assert isinstance(result, list)
    assert len(result) == total_wc // min_wc
    assert sum(result) == total_wc
    for wc in result:
        assert wc == min_wc


def test_chunk_preference_parameter():
    """Test different chunk_count_pref values."""
    total_wc = 200
    min_wc = 10
    max_wc = 50
    
    # Minimum number of chunks (chunk_count_pref = 0.0)
    chunk_count_pref = 0.0
    expected_min_chunks = math.ceil(total_wc / max_wc)  # ceil(200/50) = 4
    result = split_wc_into_chunks(total_wc, min_wc, max_wc, chunk_count_pref)
    assert isinstance(result, list)
    assert len(result) == expected_min_chunks
    assert sum(result) == total_wc
    for wc in result:
        assert min_wc <= wc <= max_wc
    
    # Maximum number of chunks (chunk_count_pref = 1.0)
    chunk_count_pref = 1.0
    expected_max_chunks = total_wc // min_wc  # 200 // 10 = 20
    result = split_wc_into_chunks(total_wc, min_wc, max_wc, chunk_count_pref)
    assert isinstance(result, list)
    assert len(result) == expected_max_chunks
    assert sum(result) == total_wc
    for wc in result:
        assert min_wc <= wc <= max_wc
    
    # Middle preference (chunk_count_pref = 0.5)
    chunk_count_pref = 0.5
    min_chunks = math.ceil(total_wc / max_wc)  # 4
    max_chunks = total_wc // min_wc  # 20
    expected_chunks = min_chunks + round(chunk_count_pref * (max_chunks - min_chunks))  # 4 + 8 = 12
    result = split_wc_into_chunks(total_wc, min_wc, max_wc, chunk_count_pref)
    assert isinstance(result, list)
    assert len(result) == expected_chunks
    assert sum(result) == total_wc
    for wc in result:
        assert min_wc <= wc <= max_wc


def test_invalid_inputs():
    """Test various invalid input cases."""
    # Invalid total_wc
    assert split_wc_into_chunks(0, 10, 20) is None  # Zero
    assert split_wc_into_chunks(-100, 10, 20) is None  # Negative
    assert split_wc_into_chunks(100.5, 10, 20) is None  # Non-integer
    
    # Invalid min/max word counts
    assert split_wc_into_chunks(100, 0, 20) is None  # min_wc = 0
    assert split_wc_into_chunks(100, 10, 0) is None  # max_wc = 0
    assert split_wc_into_chunks(100, 20, 10) is None  # min_wc > max_wc
    assert split_wc_into_chunks(100, 10.5, 20) is None  # Non-integer min_wc
    assert split_wc_into_chunks(100, 10, 20.5) is None  # Non-integer max_wc
    
    # Invalid chunk_count_pref
    assert split_wc_into_chunks(100, 10, 20, -0.1) is None  # Negative
    assert split_wc_into_chunks(100, 10, 20, 1.1) is None  # Above 1
    assert split_wc_into_chunks(100, 10, 20, chunk_count_pref="abc") is None  # Non-float
    
    # Impossible partitioning
    assert split_wc_into_chunks(50, 26, 40) is None  # min_chunks > max_chunks


def test_consistency_properties():
    """Test consistency properties across multiple runs."""
    total_wc = 500
    min_wc = 20
    max_wc = 80
    
    for _ in range(5):  # Reduced number of iterations for faster testing
        result = split_wc_into_chunks(total_wc, min_wc, max_wc)
        assert isinstance(result, list)
        # Sum consistency
        assert sum(result) == total_wc
        # Bounds consistency
        for wc in result:
            assert min_wc <= wc <= max_wc


def test_allocate_extras_function():
    """Test the _allocate_extras helper function."""
    # Basic test
    num_chunks = 5
    R = 50
    extra_max = 15
    dirichlet_a = 5.0
    extras = _allocate_extras(num_chunks, R, extra_max, dirichlet_a)
    assert len(extras) == num_chunks
    assert sum(extras) == R
    for extra in extras:
        assert 0 <= extra <= extra_max
    
    # Zero R case
    R = 0
    extras = _allocate_extras(num_chunks, R, extra_max, dirichlet_a)
    assert len(extras) == num_chunks
    assert sum(extras) == R
    assert all(e == 0 for e in extras)
    
    # Tight maximum case
    num_chunks = 10
    R = 95
    extra_max = 10
    dirichlet_a = 1.0  # More variance
    extras = _allocate_extras(num_chunks, R, extra_max, dirichlet_a)
    assert len(extras) == num_chunks
    assert sum(extras) == R
    for extra in extras:
        assert 0 <= extra <= extra_max
    
    # Full allocation case
    num_chunks = 4
    R = 40
    extra_max = 10
    dirichlet_a = 100.0
    extras = _allocate_extras(num_chunks, R, extra_max, dirichlet_a)
    assert len(extras) == num_chunks
    assert sum(extras) == R
    for extra in extras:
        assert 0 <= extra <= extra_max