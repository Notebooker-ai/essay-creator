# essay-creator

An [Open Notebook](https://open-notebook.ai) **creator** plugin: turns notebook
content into a thesis-driven **essay**, rendered by Quarto to a self-contained HTML
page plus a downloadable PDF.

- Emits the `essay.v1` artifact schema (substance in `CreationResult.files`; PDF via the `tectonic` LaTeX engine).
- Variants: **argumentative** (takes a position), **expository** (neutral explanation), **comparative** (A vs B).
- Implements the [`open-notebook-creator-sdk`](https://github.com/Notebooker-ai/open-notebook-creator-sdk) `BaseCreator` contract; registers under `open_notebook.creators`.

## Requirements

- The [`quarto`](https://quarto.org) CLI must be installed on the server (PDF uses the bundled `tectonic` engine).

## Model roles

| role | kind | requires |
|------|------|----------|
| `text` | language | `structured_json` |

## Config

| field | default | notes |
|-------|---------|-------|
| `variant` | "argumentative" | argumentative / expository / comparative |
| `length` | "medium" | short / medium / long |
| `formats` | ["html","pdf"] | output formats |

## Dev

```bash
uv sync --extra dev
uv run pytest
```

MIT licensed.
