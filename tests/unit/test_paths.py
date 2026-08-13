from pathlib import Path

from rri.utils.paths import ensure_dir, find_project_root


def test_find_project_root_locates_pyproject(tmp_path: Path) -> None:
    nested = tmp_path / "a" / "b" / "c"
    nested.mkdir(parents=True)
    (tmp_path / "pyproject.toml").touch()

    root = find_project_root(nested)

    assert root == tmp_path


def test_find_project_root_raises_when_no_marker(tmp_path: Path) -> None:
    isolated = tmp_path / "no_markers_here"
    isolated.mkdir()

    try:
        find_project_root(isolated)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError when no root marker exists above `start`")


def test_ensure_dir_creates_nested_missing_dirs(tmp_path: Path) -> None:
    target = tmp_path / "one" / "two" / "three"

    result = ensure_dir(target)

    assert result == target
    assert target.is_dir()


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "already_exists"
    target.mkdir()

    result = ensure_dir(target)

    assert result == target
    assert target.is_dir()
