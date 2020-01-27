""" Test error message string formatters."""

from epi.error_formatters import *
from pytest import raises


def test_format_type_err_msg():
    """Test that TypeError formatted strings are correct."""
    x = 20
    s1 = "foo"
    s2 = "bar"
    d = {"x": x, "s1": s1, "s2": s2}
    assert (
        format_type_err_msg(x, s1, s2, int) == "int argument foo must be int not str."
    )
    assert (
        format_type_err_msg(d, s2, x, str) == "dict argument bar must be str not int."
    )
    assert (
        format_type_err_msg(s1, s2, x, dict) == "str argument bar must be dict not int."
    )

    with raises(ValueError):
        format_type_err_msg(d, s1, s2, str)

    with raises(ValueError):
        format_type_err_msg(d, s1, x, int)

    return None


if __name__ == "__main__":
    test_format_type_err_msg()
