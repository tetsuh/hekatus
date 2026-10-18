"""Collection guard for the review-validation marker boundary."""

import pytest

import tests.conftest as repository_conftest


@pytest.mark.parametrize(
    "source",
    [
        pytest.param(
            """
            import pytest

            pytestmark = pytest.mark.slow

            def test_example():
                pass
            """,
            id="module-pytestmark",
        ),
        pytest.param(
            """
            import pytest

            class TestExample:
                pytestmark = pytest.mark.slow

                def test_example(self):
                    pass
            """,
            id="class-pytestmark",
        ),
        pytest.param(
            """
            import pytest

            @pytest.mark.slow
            class TestExample:
                def test_example(self):
                    pass
            """,
            id="slow-class-decorator",
        ),
        pytest.param(
            """
            import pytest as pytest_api

            pytestmark = pytest_api.mark.slow

            def test_example():
                pass
            """,
            id="module-pytestmark-import-alias",
        ),
    ],
)
def test_module_and_class_slow_markers_fail_collection(pytester, source):
    pytester.makepyfile(test_sample=source)

    result = pytester.runpytest_inprocess("-q", plugins=[repository_conftest])

    assert result.ret == pytest.ExitCode.USAGE_ERROR
    output = "\n".join((*result.stdout.lines, *result.stderr.lines))
    assert "slow marker is only allowed on individual tests" in output


def test_function_slow_marker_passes_collection(pytester):
    pytester.makepyfile(
        test_sample="""
        import pytest

        @pytest.mark.slow
        def test_function_slow():
            pass
        """
    )

    result = pytester.runpytest_inprocess("-q", plugins=[repository_conftest])

    result.assert_outcomes(passed=1)


def test_parametrized_slow_marker_passes_collection(pytester):
    pytester.makepyfile(
        test_sample="""
        import pytest

        @pytest.mark.parametrize(
            "value",
            [pytest.param(1, marks=pytest.mark.slow)],
        )
        def test_parametrized_slow(value):
            assert value == 1
        """
    )

    result = pytester.runpytest_inprocess("-q", plugins=[repository_conftest])

    result.assert_outcomes(passed=1)
