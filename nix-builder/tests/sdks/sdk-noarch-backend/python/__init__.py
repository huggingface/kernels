def cdiv(a: int, b: int) -> int:
    """Ceiling division for a non-negative `a` and a positive `b`."""
    if a < 0 or b <= 0:
        raise ValueError("cdiv requires a >= 0 and b > 0")
    return (a + b - 1) // b


def round_up(x: int, multiple: int) -> int:
    """Round a non-negative `x` up to a multiple of a positive `multiple`."""
    return cdiv(x, multiple) * multiple
