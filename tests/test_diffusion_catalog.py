"""The curated checkpoint catalog.

Entries are single .safetensors files, the artifact every other tool in this
ecosystem uses. Diffusers folder layouts are deliberately unsupported.
"""
import io
import os

from utils import diffusion_catalog as cat


def entry(**overrides):
    base = dict(name="X", style="s", base_model="SD 1.5",
                filename="x.safetensors", url="https://host/x.safetensors",
                size_mb="2038", author="a", license="l", trigger_words="")
    base.update(overrides)
    return base


HEADER = "name,style,base_model,filename,url,size_mb,author,license\n"


def test_shipped_catalog_parses_and_is_well_formed():
    rows = cat.load_catalog()
    assert rows, "the bundled catalog must not be empty"
    for row in rows:
        assert row["filename"].endswith(".safetensors")
        assert "/" not in row["filename"] and "\\" not in row["filename"]
        assert row["url"].startswith("https://")
        int(row["size_mb"])


def test_malformed_rows_are_skipped_not_fatal():
    rows = cat._parse(io.StringIO(
        HEADER + "A,s,SD 1.5,a.safetensors,https://h/a,10,a,l\n,,,,,,,\n"))
    assert len(rows) == 1 and rows[0]["name"] == "A"


def test_a_row_that_is_not_safetensors_is_skipped():
    """Pickles execute arbitrary code on load, so they cannot be listed."""
    rows = cat._parse(io.StringIO(HEADER + "A,s,SD 1.5,a.ckpt,https://h/a,10,a,l\n"))
    assert rows == []


def test_destination_is_a_file_directly_in_the_checkpoints_folder():
    """No folder per model: a checkpoint sits beside the others, like a LoRA."""
    dest = cat.destination(entry(), os.path.join("root", "checkpoints"))
    assert dest == os.path.join("root", "checkpoints", "x.safetensors")


def test_installed_check_tracks_the_file(tmp_path):
    e = entry()
    assert not cat.is_installed(e, str(tmp_path))
    (tmp_path / "x.safetensors").write_bytes(b"")
    assert cat.is_installed(e, str(tmp_path))


def test_a_partial_download_does_not_count_as_installed(tmp_path):
    """download_file writes to .part and renames, so a broken transfer leaves
    the checkpoint absent rather than present and truncated."""
    e = entry()
    (tmp_path / "x.safetensors.part").write_bytes(b"half")
    assert not cat.is_installed(e, str(tmp_path))


def test_every_catalog_entry_lands_where_the_dropdown_looks(tmp_path, monkeypatch):
    """Ties the catalog to the listing: a download the dropdown cannot see is
    the bug that shipped when checkpoints were folders."""
    from utils import model_dir
    monkeypatch.setattr(model_dir, "diffusion_checkpoints_dir", lambda: str(tmp_path))
    for e in cat.load_catalog():
        open(cat.destination(e, str(tmp_path)), "wb").close()
    listed = {os.path.basename(p) for p in model_dir.list_diffusion_checkpoints()}
    assert listed == {e["filename"] for e in cat.load_catalog()}
