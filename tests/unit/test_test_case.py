import numpy as np
import pytest

from tsbk.test_case import TestCase


@pytest.mark.parametrize(
    "output, expected, allow_nan, allow_inf, rtol, atol",
    [
        (np.array([1.0]), np.array([1.0]), False, False, 1.0e-5, 1.0e-8),
        (np.array([1.0]), np.array([1.0000000001]), False, False, 1.0e-5, 1.0e-8),
        (np.array(["value"], dtype=object), np.array(["value"], dtype=object), False, False, 1.0e-5, 1.0e-8),
        (np.array([np.nan]), np.array([np.nan]), True, False, 1.0e-5, 1.0e-8),
        (np.array([np.inf]), np.array([np.inf]), False, True, 1.0e-5, 1.0e-8),
    ],
)
def test_validate_output_success(output, expected, allow_nan, allow_inf, rtol, atol):
    TestCase.validate_output(output, expected, "test", allow_nan=allow_nan, allow_inf=allow_inf, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "output, expected, allow_nan, allow_inf, rtol, atol",
    [
        (np.array([1.0]), np.array([2.0]), False, False, 1.0e-5, 1.0e-8),
        (np.array([1.0]), np.array([1.0000000001]), False, False, 1.0e-20, 1.0e-20),
        (np.array([1.0]), np.array([1.0000000001]), False, False, 0, 0),
        (np.array(["value1"], dtype=object), np.array(["value2"], dtype=object), False, False, 1.0e-5, 1.0e-8),
        (np.array([np.nan]), np.array([np.nan]), False, False, 1.0e-5, 1.0e-8),
        (np.array([np.inf]), np.array([np.inf]), False, False, 1.0e-5, 1.0e-8),
    ],
)
def test_validate_output_fail(output, expected, allow_nan, allow_inf, rtol, atol):
    with pytest.raises(ValueError):
        TestCase.validate_output(
            output, expected, "test", allow_nan=allow_nan, allow_inf=allow_inf, rtol=rtol, atol=atol
        )


def test_validate_output_relaxed_float_dtypes_allows_float_mismatch():
    output = np.array([1.0], dtype=np.float16)
    expected = np.array([1.0], dtype=np.float32)
    TestCase.validate_output(output, expected, "test", relaxed_float_dtypes=True)


def test_validate_output_relaxed_float_dtypes_false_rejects_float_mismatch():
    output = np.array([1.0], dtype=np.float16)
    expected = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="incorrect dtype"):
        TestCase.validate_output(output, expected, "test", relaxed_float_dtypes=False)


def test_validate_output_relaxed_float_dtypes_rejects_non_float_mismatch():
    output = np.array([1], dtype=np.int32)
    expected = np.array([1.0], dtype=np.float32)
    with pytest.raises(ValueError, match="incorrect dtype"):
        TestCase.validate_output(output, expected, "test", relaxed_float_dtypes=True)
