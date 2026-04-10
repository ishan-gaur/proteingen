# Docs — Agent Notes

MkDocs Material documentation site for ProtStar.

## Build & Serve

- Build: `uv run mkdocs build` — output in `site/`
- Serve: `uv run mkdocs serve` — served at `/protstar/` prefix (from `site_url`), not root `/`
- `site/` is in `.gitignore`

## Theme & Extensions

- MkDocs Material with extensions:
  - `pymdownx.tabbed` (alternate_style) — content tabs via `=== "Tab Name"` (indent 4 spaces)
  - `pymdownx.highlight` (anchor_linenums) — code highlighting with `` ```python hl_lines="2 3" ``
  - `pymdownx.superfences`, `admonition`, `attr_list`
- True side-by-side columns (CSS grid) not feasible — content tabs are the idiomatic alternative

## API Reference

- `mkdocstrings[python]` configured with `paths: [src]` — resolves `protstar.*` modules
- API reference pages are thin stubs with `::: protstar.<module>` directives
- griffe warnings about missing type annotations in `generative_modeling.py` (line 152 `**kwargs`, line 408 return) — harmless, fix by adding annotations

## Nav Structure

Home | Setup | Examples | Models | Workflows (Overview, ProteinGuide) | Contributing | Reference (Design Philosophy, API Reference per-module)

- Workflow sub-pages are stubs with "Coming soon" admonitions and TODO[pi] markers
- `PLAN.md` at repo root documents full docs roadmap

## Documentation Writing Style

Learned from user's revision of `index.md`. Two tones depending on section:

### Warm/approachable pages: Home (`index.md`), Setup, Examples

- **Audience-first framing** — lead with what the user gets, not what the library does internally. Open with concrete deliverables (e.g. numbered list of capabilities), not abstract architecture descriptions. This applies at the sentence level too: when presenting a capability, lead with the reader's situation first ("If there's a model you want, all you have to do is ask"), not just the tool.
- **Address both audiences explicitly** — computational and wet-lab readers have different value props. Acknowledge both, bridge with warmth ("Let's engineer some amazing new proteins together!"). Be specific about referents — "our wet-lab collaborators" not just "our collaborators". Every noun is a chance to reinforce messaging.
- **Personal lab voice** — establish credibility through shared experience ("we use this with our wet-lab collaborators", "the same checklists we use"), not detached marketing claims. Use "research" not "lab" when referring to computational work — "lab" can imply wet-lab.
- **Benefit-oriented section titles** — "Switching Models and Algorithms Made Easy" not "Trying new models is dead simple". Professional but approachable, not slangy.
- **Code examples: most relatable case first** — lead with the scenario closest to what the paper/workflow actually did, not the flashiest option. Add inline comments explaining what each model is (e.g. `# inverse-folding model`, `# ddg predictor trained on the Megascale dataset`).
- **Short paragraphs** — break up dense technical text. One idea per paragraph. Leave breathing room.
- **Natural sentence rhythm over compression** — don't compress into dangling participial clauses ("and when working with") when a full independent clause reads better ("and we use it with ... as well"). Conciseness shouldn't sacrifice flow.
- **Inviting call-to-action sections** — frame contributions and extensibility as welcoming invitations ("No problem!", "We'd love to include your work, even if you've never contributed to open source before!").

### Technical/precise pages: Workflows, Contributing, Reference (Design Philosophy, API)

- **Clear technical language** — not salesy. These readers already bought in; they need accurate, concise explanations.
- **Describe what things do and why** — explain algorithms, abstractions, and design choices directly. No hype framing.
- **Precise terminology** — use proper ML/protein terms without hedging. Assume the reader knows what a logit, a masked language model, or inverse folding is.
- **Structure for reference use** — readers will scan and revisit. Use consistent heading hierarchies, definition lists, and code blocks they can copy-paste.
- **Still short paragraphs** — clarity applies everywhere. But the tone is matter-of-fact, not warm.

### Shared across all pages

- **Don't delete displaced text** — move to HTML comments with `TODO[pi]` noting where it should eventually go. Preserve ideas for later placement.
- **No quick-links navigation bars** at the top of pages — let the content and sidebar handle navigation.

## Known Issues

- `index.md` link `ishangaur.com` flagged as broken (missing `https://` prefix)
