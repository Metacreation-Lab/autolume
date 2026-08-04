"""The curated checkpoint catalog and how entries resolve to downloadable files."""
import io
import os

import pytest

from utils import diffusion_catalog as cat


class FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, files):
        self.files = files

    def get(self, url, timeout=None):
        return FakeResponse({"siblings": [{"rfilename": f} for f in self.files]})


HF_FILES = ["model_index.json",
            "unet/config.json",
            "unet/diffusion_pytorch_model.safetensors",
            "unet/diffusion_pytorch_model.fp16.safetensors",
            "unet/diffusion_pytorch_model.bin",
            "vae/diffusion_pytorch_model.safetensors",
            "vae/diffusion_pytorch_model.fp16.safetensors",
            "text_encoder/model.bin",
            "text_encoder/model.safetensors",
            "tokenizer/vocab.json",
            "tokenizer/merges.txt",
            "unet/diffusion_pytorch_model.non_ema.safetensors",
            "model.onnx"]


def entry(**overrides):
    base = dict(name="X", style="s", base_model="SD 1.5", source="hf",
                ref="owner/model", dest="model", variant="", size_mb="2100",
                author="a", license="l", trigger_words="", rating="-")
    base.update(overrides)
    return base


def test_shipped_catalog_parses_and_is_well_formed():
    rows = cat.load_catalog()
    assert rows, "the bundled catalog must not be empty"
    for row in rows:
        assert row["source"] in ("hf", "file")
        assert row["dest"] and "/" not in row["dest"] and "\\" not in row["dest"]
        assert row["variant"] in ("", "fp16")
        int(row["size_mb"])  # must be a number


def test_malformed_rows_are_skipped_not_fatal():
    good = "name,style,base_model,source,ref,dest,size_mb,author,license\n"
    rows = cat._parse(io.StringIO(good + "A,s,SD 1.5,hf,o/m,m,10,a,l\n,,,,,,,,\n"))
    assert len(rows) == 1 and rows[0]["name"] == "A"


def test_unknown_source_is_skipped():
    header = "name,style,base_model,source,ref,dest,size_mb,author,license\n"
    rows = cat._parse(io.StringIO(header + "A,s,SD 1.5,ftp,o/m,m,10,a,l\n"))
    assert rows == []


def test_no_pickle_ever_reaches_the_download_list():
    files = cat.resolve_files(entry(), session=FakeSession(HF_FILES))
    urls = [u for u, _d in files]
    for banned in (".bin", ".ckpt", ".pth", ".onnx", ".non_ema"):
        assert not any(banned in u for u in urls), banned


def test_default_variant_takes_the_plain_weights():
    files = cat.resolve_files(entry(), session=FakeSession(HF_FILES))
    names = [d for _u, d in files]
    assert os.path.join("model", "unet", "diffusion_pytorch_model.safetensors") in names
    assert not any("fp16" in n for n in names)


def test_fp16_variant_takes_half_precision_and_stores_it_unsuffixed():
    files = cat.resolve_files(entry(variant="fp16"), session=FakeSession(HF_FILES))
    urls = [u for u, _d in files]
    names = [d for _u, d in files]
    # fetched from the fp16 file...
    assert any("diffusion_pytorch_model.fp16.safetensors" in u for u in urls)
    # ...but stored plain, because from_pretrained only looks for the plain name
    # and the wrapper gives us no way to pass variant="fp16" through
    assert not any("fp16" in n for n in names)
    assert os.path.join("model", "unet", "diffusion_pytorch_model.safetensors") in names


def test_the_redundant_root_checkpoint_is_not_downloaded():
    """Diffusers repos often also ship the whole model as one root-level file.
    Fetching it alongside the folder layout doubles the download for nothing."""
    files = cat.resolve_files(entry(), session=FakeSession(HF_FILES + ["whole_model.safetensors"]))
    names = [d for _u, d in files]
    assert os.path.join("model", "whole_model.safetensors") not in names
    assert os.path.join("model", "unet", "diffusion_pytorch_model.safetensors") in names


def test_a_single_file_repo_still_yields_its_weights():
    """Without model_index.json the root file is the model, not a duplicate."""
    files = cat.resolve_files(entry(), session=FakeSession(["config.json", "model.safetensors"]))
    assert [d for _u, d in files if d.endswith(".safetensors")] == \
        [os.path.join("model", "model.safetensors")]


def test_a_pickle_only_repo_is_rejected_loudly():
    pickle_only = ["model_index.json", "unet/diffusion_pytorch_model.bin"]
    with pytest.raises(ValueError, match="must not be in the catalog"):
        cat.resolve_files(entry(), session=FakeSession(pickle_only))


def test_fp16_requested_but_absent_is_rejected():
    plain_only = ["model_index.json", "unet/diffusion_pytorch_model.safetensors"]
    with pytest.raises(ValueError):
        cat.resolve_files(entry(variant="fp16"), session=FakeSession(plain_only))


def test_single_file_entries_resolve_to_one_download():
    e = entry(source="file", ref="https://host/x.safetensors", dest="x.safetensors")
    assert cat.resolve_files(e) == [("https://host/x.safetensors", "x.safetensors")]


def test_a_half_finished_folder_does_not_count_as_installed(tmp_path):
    e = entry()
    root = tmp_path / "model"
    root.mkdir()
    (root / "model_index.json").write_text("{}")
    # index arrived before the weights: loadable-looking, not loadable
    assert not cat.is_installed(e, str(tmp_path))
    (root / "unet").mkdir()
    assert cat.is_installed(e, str(tmp_path))


def test_single_file_installed_check(tmp_path):
    e = entry(source="file", ref="https://host/x.safetensors", dest="x.safetensors")
    assert not cat.is_installed(e, str(tmp_path))
    (tmp_path / "x.safetensors").write_bytes(b"")
    assert cat.is_installed(e, str(tmp_path))


def test_listing_finds_both_files_and_diffusers_folders(tmp_path, monkeypatch):
    """A HuggingFace model is a directory, so a file-only listing would make
    every catalog download invisible in the dropdown."""
    from utils import model_dir
    monkeypatch.setattr(model_dir, "diffusion_checkpoints_dir", lambda: str(tmp_path))
    (tmp_path / "single.safetensors").write_bytes(b"")
    (tmp_path / "notamodel").mkdir()
    repo = tmp_path / "adiffusersrepo"
    (repo / "unet").mkdir(parents=True)
    (repo / "model_index.json").write_text("{}")
    found = [os.path.basename(p) for p in model_dir.list_diffusion_checkpoints()]
    assert found == ["adiffusersrepo", "single.safetensors"]
    assert "notamodel" not in found
