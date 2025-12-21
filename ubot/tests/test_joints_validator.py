import pytest

from ubot.core.validator import ValidationError, validate_joints


def test_valid_joints_opposite():
    validate_joints({0: [90, 90], 1: [-90, -90]}, modules=[0, 1])


def test_valid_joints_zero_flexibility():
    validate_joints({0: [0, 90], 1: [-90, 0]}, modules=[0, 1])


def test_invalid_value():
    with pytest.raises(ValidationError):
        validate_joints({0: [30, 0]}, modules=[0])


def test_invalid_not_opposite():
    with pytest.raises(ValidationError):
        validate_joints({0: [90, -90]}, modules=[0])


def test_missing_module():
    with pytest.raises(ValidationError):
        validate_joints({}, modules=[0])
