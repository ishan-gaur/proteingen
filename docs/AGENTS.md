# Docs — Agent Notes

MkDocs Material documentation site for ProteinGen.

## Build & Serve

- Build: `uv run mkdocs build` — output in `site/`
- Serve: `uv run mkdocs serve` — served at `/proteingen/` prefix (from `site_url`), not root `/`
- `site/` is in `.gitignore`

## Theme & Extensions

- MkDocs Material with extensions:
  - `pymdownx.tabbed` (alternate_style) — content tabs via `=== "Tab Name"` (indent 4 spaces)
  - `pymdownx.highlight` (anchor_linenums) — code highlighting with `` ```python hl_lines="2 3" ``
  - `pymdownx.superfences`, `admonition`, `attr_list`
- True side-by-side columns (CSS grid) not feasible — content tabs are the idiomatic alternative

## API Reference

- `mkdocstrings[python]` configured with `paths: [src]` — resolves `proteingen.*` modules
- API reference pages are thin stubs with `::: proteingen.<module>` directives
- griffe warnings about missing type annotations in `generative_modeling.py` (line 152 `**kwargs`, line 408 return) — harmless, fix by adding annotations

## Nav Structure

Home | Setup | Examples | Models | Workflows (Overview, ProteinGuide) | Contributing | Reference (Design Philosophy, API Reference per-module)

- Workflow sub-pages are stubs with "Coming soon" admonitions and TODO[pi] markers
- `PLAN.md` at repo root documents full docs roadmap

## Known Issues

- `index.md` link `ishangaur.com` flagged as broken (missing `https://` prefix)
