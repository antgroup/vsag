# VSAG landing page (vsag.io)

Static marketing / landing page served at the root of
[vsag.io](https://vsag.io/).

## Files

- `index.html` — single-page landing
- `style.css` — brand-aligned styles (gold gradient from the project logo,
  warm-graphite neutrals, system fonts, dark mode via `prefers-color-scheme`)
- `favicon.svg` — shield + V glyph derived from the project logo

No build step, no JavaScript, no external dependencies.

## Deployment

Changes to any file under `docs/landing/` are auto-synced to the
[`vsag-io/vsag-io.github.io`](https://github.com/vsag-io/vsag-io.github.io)
repository (branch `sync`) by the workflow
[`.github/workflows/docs.yaml`](../../.github/workflows/docs.yaml), alongside
the English / Chinese mdbook docs and blog.

On the target repo the files land under `landing/`; the Pages site is
responsible for mapping `landing/index.html` to `/` (and `landing/style.css` to
`/style.css`, etc.).

## Local preview

```bash
cd docs/landing
python3 -m http.server 8000
# open http://localhost:8000/
```

## Editing

- Keep content factual and in sync with `docs/docs/en/src/`. If a capability
  changes materially, update the relevant section here.
- Brand colours come from `docs/banner.svg`: gold gradient
  `#fce897` → `#f4bd37`, with graphite text `#171412`.
- Keep the page dependency-free; do not add runtime JS or CDN assets.
