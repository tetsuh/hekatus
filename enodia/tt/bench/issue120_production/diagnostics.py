"""Phase-specific diagnostics, never inputs to coverage success."""


def expectations(consumed: int, configured: int) -> dict[str, int]:
    if type(consumed) is not int or type(configured) is not int or not 0 <= consumed <= configured <= 500000:
        raise ValueError("invalid diagnostic frame accounting")
    return {"expected_after_consumed_frames": consumed + 2,
            "expected_after_full_run": configured + 2}
