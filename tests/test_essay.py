"""generate() tests with a fake LLM. The Quarto render runs for real when `quarto`
is on PATH; otherwise we still assert the .qmd was assembled/sanitized and that the
creator fails gracefully.
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

import pytest
from open_notebook_creator_sdk import ContentBundle, CreationRequest, ModelRole
from open_notebook_creator_sdk.testing import assert_creator_compliant

from essay_creator import EssayCreator

HAS_QUARTO = shutil.which("quarto") is not None


class _FakeLLM:
    def __init__(self, content: str):
        self._content = content

    async def ainvoke(self, _prompt):
        return type("Resp", (), {"content": self._content})()


class _FakeRole(ModelRole):
    """Returns a JSON outline for the structured call, markdown for section calls."""

    def create_language(self, **kwargs):
        if "structured" in kwargs:  # outline call
            return _FakeLLM(
                json.dumps(
                    {
                        "title": "On Photosynthesis",
                        "thesis": "Photosynthesis is the engine of life on Earth.",
                        "sections": [
                            {"title": "Introduction", "summary": "Frames the claim."},
                            {"title": "The Mechanism", "summary": "Light to sugar."},
                        ],
                    }
                )
            )
        return _FakeLLM("Some **prose** 🎉 with a blank ____________________ here.\n")


def _request(output_dir: str, formats=None, variant="argumentative") -> CreationRequest:
    return CreationRequest(
        content=ContentBundle(text="Photosynthesis converts light to chemical energy.", token_count=8),
        config={"variant": variant, "length": "short", "formats": formats or ["html"]},
        models={"text": _FakeRole(provider="fake", model="fake")},
        output_dir=output_dir,
        artifact_id="artifact-test-1",
    )


def test_static_compliance():
    assert_creator_compliant(EssayCreator())


@pytest.mark.asyncio
async def test_assembles_and_sanitizes_qmd():
    with tempfile.TemporaryDirectory() as d:
        await EssayCreator().generate(_request(d, ["html"]))
        qmd = (Path(d) / "essay.qmd").read_text(encoding="utf-8")
        assert 'title: "On Photosynthesis"' in qmd
        assert "abstract:" in qmd  # thesis carried into front matter
        assert "## Introduction" in qmd and "## The Mechanism" in qmd
        assert "pdf-engine: tectonic" not in qmd  # pdf not requested
        # sanitizer applied
        assert "🎉" not in qmd
        assert "____________________" not in qmd and "__________" in qmd


@pytest.mark.asyncio
async def test_no_text_role_is_failure():
    with tempfile.TemporaryDirectory() as d:
        req = CreationRequest(content=ContentBundle(text="x"), output_dir=d, artifact_id="a")
        result = await EssayCreator().generate(req)
        assert result.status == "FAILURE"
        assert result.errors[0].phase == "setup"


@pytest.mark.skipif(not HAS_QUARTO, reason="quarto not installed")
@pytest.mark.asyncio
async def test_renders_html():
    with tempfile.TemporaryDirectory() as d:
        result = await EssayCreator().generate(_request(d, ["html"]))
        assert result.status == "SUCCESS", (result.user_message, result.errors)
        assert result.schema_id == "essay.v1"
        assert result.data["formats"] == ["html"]
        assert result.data["variant"] == "argumentative"
        assert (Path(d) / "essay.html").stat().st_size > 0
        assert result.files[0].content_type == "text/html"
