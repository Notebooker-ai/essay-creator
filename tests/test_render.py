"""Post-generation editing: generate() attaches the .qmd as a ``source`` file and
render() re-runs Quarto on an edited .qmd without touching the LLM. ``_quarto_render``
is monkeypatched to a fake that writes the expected output file so the tests do not
depend on the ``quarto`` binary.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from open_notebook_creator_sdk import RenderRequest, SourceDoc
from test_essay import _request

import essay_creator
from essay_creator import EssayCreator

EDITED_QMD = (
    '---\ntitle: "Edited Essay"\nformat:\n  html:\n    embed-resources: true\n---\n\n'
    "## Introduction\n\nHand-edited section.\n"
)


def _fake_quarto(rendered: list[str], fail: set[str] | None = None):
    """Fake ``_quarto_render``: records calls and writes ``essay.<fmt>``."""

    async def fake(output_dir: Path, fmt: str) -> None:
        assert (output_dir / "essay.qmd").exists()
        rendered.append(fmt)
        if fail and fmt in fail:
            raise RuntimeError(f"boom {fmt}")
        (output_dir / f"essay.{fmt}").write_text(f"<{fmt}>", encoding="utf-8")

    return fake


def _render_request(output_dir: str, sources: list[SourceDoc], formats: list[str]) -> RenderRequest:
    return RenderRequest(
        sources=sources,
        config={"variant": "expository", "length": "short", "formats": formats},
        data={
            "title": "On Photosynthesis",
            "thesis": "Photosynthesis is the engine of life on Earth.",
            "variant": "expository",
            "formats": ["html"],
        },
        output_dir=output_dir,
        artifact_id="artifact-test-1",
    )


def test_manifest_declares_editable_source():
    assert EssayCreator().manifest.editable_source is True


@pytest.mark.asyncio
async def test_generate_attaches_qmd_as_source(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(essay_creator, "_quarto_render", _fake_quarto(calls))
    with tempfile.TemporaryDirectory() as d:
        result = await EssayCreator().generate(_request(d, ["pdf", "html"]))
        assert result.status == "SUCCESS", (result.user_message, result.errors)
        sources = [f for f in result.files if f.role == "source"]
        assert len(sources) == 1
        src = sources[0]
        assert src.filename == "essay.qmd" and src.path == "essay.qmd"
        assert src.content_type == "text/markdown"
        assert (Path(d) / src.path).exists()
        outputs = [f for f in result.files if f.role == "output"]
        # Outputs come first, HTML first among them; the source trails.
        assert [f.filename for f in outputs] == ["essay.html", "essay.pdf"]
        assert result.files[0].content_type == "text/html"
        assert result.files[-1].role == "source"


@pytest.mark.asyncio
async def test_render_rerenders_edited_qmd(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(essay_creator, "_quarto_render", _fake_quarto(calls))
    with tempfile.TemporaryDirectory() as d:
        req = _render_request(
            d,
            [SourceDoc(filename="essay.qmd", content_type="text/markdown", content=EDITED_QMD)],
            ["html", "pdf"],
        )
        result = await EssayCreator().render(req)
        assert result.status == "SUCCESS", (result.user_message, result.errors)
        assert result.schema_id == "essay.v1"
        assert calls == ["html", "pdf"]
        assert (Path(d) / "essay.qmd").read_text(encoding="utf-8") == EDITED_QMD
        outputs = [f for f in result.files if f.role == "output"]
        assert [f.filename for f in outputs] == ["essay.html", "essay.pdf"]
        assert result.files[0].content_type == "text/html"
        assert [f.filename for f in result.files if f.role == "source"] == ["essay.qmd"]
        # data: formats reflect this render; everything else carried forward.
        assert result.data["formats"] == ["html", "pdf"]
        assert result.data["title"] == "On Photosynthesis"
        assert result.data["thesis"] == "Photosynthesis is the engine of life on Earth."
        assert result.data["variant"] == "expository"


@pytest.mark.asyncio
async def test_render_partial_when_one_format_fails(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(essay_creator, "_quarto_render", _fake_quarto(calls, fail={"pdf"}))
    with tempfile.TemporaryDirectory() as d:
        req = _render_request(
            d,
            [SourceDoc(filename="essay.qmd", content_type="text/markdown", content=EDITED_QMD)],
            ["html", "pdf"],
        )
        result = await EssayCreator().render(req)
        assert result.status == "PARTIAL"
        assert result.data["formats"] == ["html"]
        assert any(e.phase == "render" and "pdf" in e.message for e in result.errors)
        assert [f.filename for f in result.files if f.role == "output"] == ["essay.html"]
        assert [f.filename for f in result.files if f.role == "source"] == ["essay.qmd"]


@pytest.mark.asyncio
async def test_render_without_qmd_source_is_failure(monkeypatch):
    calls: list[str] = []
    monkeypatch.setattr(essay_creator, "_quarto_render", _fake_quarto(calls))
    with tempfile.TemporaryDirectory() as d:
        req = _render_request(
            d,
            [SourceDoc(filename="other.md", content_type="text/markdown", content="# nope")],
            ["html"],
        )
        result = await EssayCreator().render(req)
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "render"
        assert "essay.qmd" in result.errors[0].message
        assert calls == []


@pytest.mark.asyncio
async def test_render_when_quarto_missing_is_failure(monkeypatch):
    async def missing(_output_dir: Path, _fmt: str) -> None:
        raise FileNotFoundError("quarto")

    monkeypatch.setattr(essay_creator, "_quarto_render", missing)
    with tempfile.TemporaryDirectory() as d:
        req = _render_request(
            d,
            [SourceDoc(filename="essay.qmd", content_type="text/markdown", content=EDITED_QMD)],
            ["html", "pdf"],
        )
        result = await EssayCreator().render(req)
        assert result.status == "FAILURE"
        assert result.errors[0].message == "quarto not installed"
