"""essay-creator: an Open Notebook creator that turns notebook content into a
thesis-driven **essay**, rendered by Quarto to a self-contained HTML page plus a
downloadable PDF (emitted as ``essay.v1``).

The LLM first plans a thesis + section outline (tuned to the chosen variant —
argumentative / expository / comparative), then writes one section at a time. The
sections are assembled into a single ``.qmd`` and handed to the ``quarto`` CLI. PDF
uses the ``tectonic`` engine so no full TeX Live install is needed.

The assembled ``.qmd`` is attached to the result as a ``source`` file so the host
can let users edit it and re-render (``render``) without another LLM pass.
"""

from __future__ import annotations

import asyncio
import json
import re
from importlib import resources
from pathlib import Path
from typing import ClassVar, List, Literal

from ai_prompter import Prompter
from loguru import logger
from open_notebook_creator_sdk import (
    BaseCreator,
    CreationError,
    CreationFile,
    CreationRequest,
    CreationResult,
    CreatorManifest,
    ModelRoleSpec,
    RenderRequest,
)
from open_notebook_creator_sdk.schemas import EssayV1
from pydantic import BaseModel, Field

from .sanitize import sanitize_markdown

__version__ = "0.2.0"

_QMD_NAME = "essay.qmd"
_QMD_STEM = "essay"

# format key -> (file extension, MIME type, UI label)
_FORMAT_META: dict[str, tuple[str, str, str]] = {
    "html": ("html", "text/html", "HTML"),
    "pdf": ("pdf", "application/pdf", "PDF"),
}

# How many body sections each length target aims for.
_SECTIONS_FOR = {"short": 3, "medium": 5, "long": 8}

# quarto's first render can be slow (tectonic may fetch packages); cap per format.
_RENDER_TIMEOUT_S = 600


class EssayConfig(BaseModel):
    """Per-generation config; its JSON Schema drives the host's generate form."""

    variant: Literal["argumentative", "expository", "comparative"] = Field(
        default="argumentative",
        description="Argumentative (takes a position), expository (neutral explanation), or comparative (A vs B).",
    )
    length: Literal["short", "medium", "long"] = Field(
        default="medium", description="Approximate essay length"
    )
    formats: List[Literal["html", "pdf"]] = Field(
        default=["html", "pdf"], description="Output formats to render"
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n", "", text)
        text = re.sub(r"\n```$", "", text)
    return text.strip()


def _render_prompt(name: str, ctx: dict) -> str:
    template = resources.files("essay_creator.prompts").joinpath(name).read_text()
    return Prompter(template_text=template).render(ctx)


def _build_front_matter(title: str, thesis: str | None, formats: List[str]) -> str:
    lines = ["---", f"title: {json.dumps(title)}"]
    if thesis:
        lines.append(f"abstract: {json.dumps(thesis)}")
    lines.append("format:")
    if "html" in formats:
        lines += ["  html:", "    embed-resources: true"]
    if "pdf" in formats:
        lines += [
            "  pdf:",
            "    pdf-engine: tectonic",
            "    geometry:",
            "      - margin=1in",
        ]
    lines.append("---")
    return "\n".join(lines)


async def _quarto_render(output_dir: Path, fmt: str) -> None:
    """Render ``essay.qmd`` to one format in-place. Raises on failure."""
    proc = await asyncio.create_subprocess_exec(
        "quarto",
        "render",
        _QMD_NAME,
        "--to",
        fmt,
        cwd=str(output_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, err = await asyncio.wait_for(proc.communicate(), timeout=_RENDER_TIMEOUT_S)
    if proc.returncode != 0:
        detail = (err.decode(errors="replace") or out.decode(errors="replace")).strip()
        raise RuntimeError(detail[-2000:] or f"quarto exited {proc.returncode}")


async def _render_formats(
    output_dir: Path, formats: list[str]
) -> tuple[list[CreationFile], list[str], list[CreationError], list[str]]:
    """Render ``essay.qmd`` (already in ``output_dir``) to each requested format.

    Best-effort: one format failing is non-fatal; a missing ``quarto`` binary
    aborts the loop. Returns ``(files, warnings, errors, rendered_formats)`` with
    the HTML output (the inline-viewable file) sorted first.
    """
    files: list[CreationFile] = []
    warnings: list[str] = []
    errors: list[CreationError] = []
    rendered: list[str] = []
    for fmt in formats:
        ext, content_type, label = _FORMAT_META[fmt]
        out_name = f"{_QMD_STEM}.{ext}"
        try:
            await _quarto_render(output_dir, fmt)
            if not (output_dir / out_name).exists():
                raise RuntimeError("quarto reported success but produced no output file")
            files.append(
                CreationFile(filename=out_name, content_type=content_type, path=out_name, label=label)
            )
            rendered.append(fmt)
        except FileNotFoundError:
            logger.error("essay: 'quarto' binary not found on PATH")
            errors.append(CreationError(phase="render", message="quarto not installed"))
            warnings.append("Quarto is not installed on the server; cannot render the essay.")
            break
        except Exception as e:  # noqa: BLE001 - one format failing is non-fatal
            logger.warning(f"essay: {fmt} render failed: {e}")
            warnings.append(f"{label} export failed.")
            errors.append(CreationError(phase="render", message=f"{fmt}: {e}"))

    # HTML (the inline-viewable file) first, then the rest.
    files.sort(key=lambda f: 0 if f.content_type == "text/html" else 1)
    return files, warnings, errors, rendered


def _source_file() -> CreationFile:
    """The ``.qmd`` as a ``role="source"`` file: the host shows it in its source
    editor and hands the edited text back through :meth:`EssayCreator.render`."""
    return CreationFile(
        filename=_QMD_NAME,
        content_type="text/markdown",
        path=_QMD_NAME,
        role="source",
        label="Source",
    )


class EssayCreator(BaseCreator):
    config_model: ClassVar[type] = EssayConfig

    @property
    def manifest(self) -> CreatorManifest:
        return self.build_manifest(
            key="essays",
            name="Essays",
            version=__version__,
            description="LLM-written thesis-driven essay rendered to HTML and PDF via Quarto.",
            sdk_compat=">=0.2,<1",
            emits=["essay.v1"],
            model_roles=[
                ModelRoleSpec(
                    key="text",
                    kind="language",
                    requires=["structured_json"],
                    description="LLM that writes the essay.",
                )
            ],
            icon="pen-line",
            editable_source=True,
            suggestion_hint=(
                "the thesis to argue and which evidence, tensions, or contrasts to "
                "build the argument on"
            ),
        )

    async def generate(self, request: CreationRequest) -> CreationResult:
        cfg = EssayConfig.model_validate(request.config)
        role = request.models.get("text")
        if role is None:
            return CreationResult(
                status="FAILURE",
                schema_id="essay.v1",
                data={},
                errors=[CreationError(phase="setup", message="missing 'text' model role")],
                user_message="No language model was provided for essay generation.",
            )

        # 1. Outline (structured JSON): title, thesis, section headings.
        outline_prompt = _render_prompt(
            "outline.jinja",
            {
                "content": request.content.text,
                "variant": cfg.variant,
                "num_sections": _SECTIONS_FOR[cfg.length],
                "instructions": request.instructions,
            },
        )
        llm = role.create_language(structured={"type": "json"}, max_tokens=2000)
        resp = await llm.ainvoke(outline_prompt)
        raw = resp.content if hasattr(resp, "content") else str(resp)
        try:
            outline = json.loads(_strip_fences(raw))
        except json.JSONDecodeError as e:
            logger.error(f"essay: outline was non-JSON: {e}")
            return CreationResult(
                status="FAILURE",
                schema_id="essay.v1",
                data={},
                errors=[CreationError(phase="parse", message=f"invalid JSON: {e}", retryable=True)],
                user_message="The model returned an unparseable outline. Please retry.",
            )

        title = (outline.get("title") or "").strip() if isinstance(outline, dict) else ""
        thesis = outline.get("thesis") if isinstance(outline, dict) else None
        thesis = thesis.strip() if isinstance(thesis, str) and thesis.strip() else None
        sections_in = outline.get("sections", []) if isinstance(outline, dict) else []
        sections_meta = [
            {"title": (s.get("title") or "").strip(), "summary": (s.get("summary") or "").strip()}
            for s in sections_in
            if isinstance(s, dict) and (s.get("title") or "").strip()
        ]
        if not title or not sections_meta:
            return CreationResult(
                status="FAILURE",
                schema_id="essay.v1",
                data={},
                errors=[CreationError(phase="generate", message="no usable outline produced")],
                user_message="No essay outline could be generated from this content.",
            )

        # 2. One body section per outline entry.
        section_llm = role.create_language(max_tokens=3000)
        bodies: list[str] = []
        for i, sec in enumerate(sections_meta, start=1):
            prompt = _render_prompt(
                "section.jinja",
                {
                    "content": request.content.text,
                    "essay_title": title,
                    "thesis": thesis,
                    "variant": cfg.variant,
                    "instructions": request.instructions,
                    "section_number": i,
                    "total_sections": len(sections_meta),
                    "section_title": sec["title"],
                    "section_summary": sec["summary"],
                    "outline": sections_meta,
                },
            )
            sresp = await section_llm.ainvoke(prompt)
            body = sresp.content if hasattr(sresp, "content") else str(sresp)
            bodies.append(sanitize_markdown(_strip_fences(body)))

        # 3. Assemble the single-document .qmd.
        output_dir = Path(request.output_dir)
        parts = [_build_front_matter(title, thesis, cfg.formats), ""]
        for sec, body in zip(sections_meta, bodies):
            parts.append(f"\n## {sec['title']}\n")
            parts.append(body)
            parts.append("")
        (output_dir / _QMD_NAME).write_text("\n".join(parts), encoding="utf-8")

        # 4. Render each requested format (best-effort: one failure -> PARTIAL).
        files, warnings, errors, rendered = await _render_formats(output_dir, cfg.formats)
        if not rendered:
            return CreationResult(
                status="FAILURE",
                schema_id="essay.v1",
                data={},
                warnings=warnings,
                errors=errors or [CreationError(phase="render", message="no formats rendered")],
                user_message="The essay could not be rendered to any format.",
            )

        # Outputs first (HTML leading), then the editable source for the host's editor.
        files.append(_source_file())

        data = EssayV1(
            title=title,
            thesis=thesis,
            variant=cfg.variant,
            formats=rendered,
        ).model_dump()

        return CreationResult(
            status="PARTIAL" if errors else "SUCCESS",
            schema_id="essay.v1",
            data=data,
            files=files,
            warnings=warnings,
            errors=errors,
        )

    async def render(self, request: RenderRequest) -> CreationResult:
        """Re-render the (possibly user-edited) ``essay.qmd``. Deterministic: no LLM."""
        cfg = EssayConfig.model_validate(request.config)
        output_dir = Path(request.output_dir)

        # The host already restored every source/asset into output_dir; write the
        # sources again anyway so render() is self-contained given only the request.
        for doc in request.sources:
            target = output_dir / doc.filename
            if not target.resolve().is_relative_to(output_dir.resolve()):
                return CreationResult(
                    status="FAILURE",
                    schema_id="essay.v1",
                    data=request.data,
                    errors=[
                        CreationError(
                            phase="render", message=f"source path escapes output_dir: {doc.filename}"
                        )
                    ],
                    user_message="The essay source could not be written.",
                )
            target.write_text(doc.content, encoding="utf-8")
        if not any(doc.filename == _QMD_NAME for doc in request.sources):
            return CreationResult(
                status="FAILURE",
                schema_id="essay.v1",
                data=request.data,
                errors=[CreationError(phase="render", message=f"missing source file {_QMD_NAME}")],
                user_message="The essay source is missing; nothing to render.",
            )

        files, warnings, errors, rendered = await _render_formats(output_dir, cfg.formats)
        if not rendered:
            return CreationResult(
                status="FAILURE",
                schema_id="essay.v1",
                data=request.data,
                warnings=warnings,
                errors=errors or [CreationError(phase="render", message="no formats rendered")],
                user_message="The essay could not be rendered to any format.",
            )

        # Carry forward what the source does not encode (title, thesis, variant) and
        # replace what this render decided. Validating against the schema makes a
        # stale/garbage previous ``data`` fail loudly instead of being persisted.
        data = EssayV1.model_validate({**request.data, "formats": rendered}).model_dump()
        files.append(_source_file())
        return CreationResult(
            status="PARTIAL" if errors else "SUCCESS",
            schema_id="essay.v1",
            data=data,
            files=files,
            warnings=warnings,
            errors=errors,
        )
