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


class LinkResponse:
    def __init__(self, disposition="", length=0):
        self.headers = {"Content-Disposition": disposition, "Content-Length": str(length)}

    def raise_for_status(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class LinkSession:
    def __init__(self, response):
        self.response = response

    def get(self, url, stream=None, timeout=None):
        return self.response


def test_a_direct_safetensors_link_needs_no_round_trip():
    e = cat.entry_from_url("https://huggingface.co/o/m/resolve/main/model.safetensors")
    assert e["filename"] == "model.safetensors"
    assert e["url"].endswith("model.safetensors")


def test_a_civitai_link_takes_its_filename_from_the_server():
    """Civitai download urls carry no filename, so the server is asked."""
    session = LinkSession(LinkResponse('attachment; filename="dreamshaper_8.safetensors"',
                                       2_136_000_000))
    e = cat.entry_from_url("https://civitai.com/api/download/models/128713", session=session)
    assert e["filename"] == "dreamshaper_8.safetensors"
    assert int(e["size_mb"]) == 2_136_000_000 // (1024 * 1024)


def test_a_link_to_something_that_is_not_safetensors_is_refused():
    session = LinkSession(LinkResponse('attachment; filename="model.ckpt"'))
    for url, kwargs in [("https://host/model.ckpt", {}),
                        ("https://civitai.com/api/download/models/1", {"session": session})]:
        try:
            cat.entry_from_url(url, **kwargs)
        except ValueError as e:
            assert "safetensors" in str(e)
        else:
            raise AssertionError(f"{url} should have been refused")


def test_a_link_that_is_not_a_link_says_so():
    for text in ("stabilityai/sd-turbo", "", "   ", "just some words"):
        try:
            cat.entry_from_url(text)
        except ValueError as e:
            assert "link" in str(e).lower()
        else:
            raise AssertionError(f"{text!r} should have been refused")


def test_a_pasted_entry_downloads_like_a_catalog_entry(tmp_path):
    e = cat.entry_from_url("https://host/x.safetensors")
    assert cat.destination(e, str(tmp_path)) == os.path.join(str(tmp_path), "x.safetensors")
    assert not cat.is_installed(e, str(tmp_path))
