# Release Notes

VSAG website release notes are maintained by `MAJOR.MINOR` series. Each series page covers the
first release and every later patch release in that line. GitHub Releases remains the source for
the complete per-patch pull request list, assets, and contributor credits.

## Release Series

- [VSAG 1.0](release_notes/v1.0.md)
  - First release: [v1.0.0](https://github.com/antgroup/vsag/releases/tag/v1.0.0),
    July 12, 2026
  - Latest patch: `v1.0.0`
  - Status: stable

Future release notes follow the same layout: `v1.1`, `v1.2`, `v2.0`, and so on. Patch releases
update their existing series page instead of creating a separate website page.

## Version and Note Grouping

Release tags use the `vMAJOR.MINOR.PATCH` form. The website groups them by `MAJOR.MINOR` so each
page can explain the full series, while GitHub Releases records the exact contents of each tag.

## Getting a Specific Version

### C++ / source

```bash
git checkout vX.Y.Z
make release
```

### Python

Check [PyPI](https://pypi.org/project/pyvsag/) for an available binding version, then install that
exact version:

```bash
pip install pyvsag==X.Y.Z
```

Binding releases may not match every core C++ tag. The repository also contains C and
Node.js/TypeScript bindings. See the corresponding release series page and repository examples for
their support and packaging state.

## Upgrade Guidance

- Read the compatibility section of the target release series before upgrading.
- When a serialization format changes, validate old artifacts with the
  [compatibility check tool](check_compatibility.md) in a staging environment.
- Roll out gradually in production and use the
  [performance evaluation tool](eval.md) to compare recall, latency, and resource use.

For complete patch-level history, see
[all VSAG releases on GitHub](https://github.com/antgroup/vsag/releases).
