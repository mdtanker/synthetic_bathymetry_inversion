from pathlib import Path

import nox

DIR = Path(__file__).parent.resolve()
PROJECT = nox.project.load_toml()

nox.needs_version = ">=2025.2.9"
nox.options.default_venv_backend = "uv|virtualenv"


@nox.session
def lint(session: nox.Session) -> None:
    """
    Run the linter.
    """
    session.install("pre-commit")
    session.run(
        "pre-commit", "run", "--all-files", "--show-diff-on-failure", *session.posargs
    )


@nox.session
def pylint(session: nox.Session) -> None:
    """
    Run Pylint.
    """
    # This needs to be installed into the package environment, and is slower
    # than a pre-commit check
    session.install("-e.", "pylint>=3.2")
    session.run("pylint", "synthetic_bathymetry_inversion", *session.posargs)


@nox.session
def style(session: nox.Session) -> None:
    """
    Run the linter and Pylint.
    """
    session.notify("lint")
    session.notify("pylint")


@nox.session
def test(session: nox.Session) -> None:
    """
    Run the unit and regular tests.
    """
    test_deps = nox.project.dependency_groups(PROJECT, "test")
    session.install("-e.", *test_deps)
    session.run("pytest", "--cov", *session.posargs)
