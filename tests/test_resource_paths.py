from pathlib import Path

from utils import resource_paths


def test_repo_root_is_repo_not_src():
    assert (resource_paths._REPO_ROOT / "pyproject.toml").is_file()
    assert resource_paths._REPO_ROOT.name != "src"


def test_resource_path_finds_repo_root_data():
    assert resource_paths.resource_path("pyproject.toml").is_file()
    assert resource_paths.resource_path("sr_models").is_dir()


def test_resource_path_finds_assets_under_src():
    font = resource_paths.resource_path("assets", "OpenSans-Regular.ttf")
    assert font.is_file()
    assert font == Path(resource_paths.__file__).resolve().parents[1] / "assets" / "OpenSans-Regular.ttf"


def test_get_version_matches_pyproject():
    import tomllib

    with open(resource_paths._REPO_ROOT / "pyproject.toml", "rb") as fp:
        expected = tomllib.load(fp)["project"]["version"]
    assert resource_paths.get_version() == expected
