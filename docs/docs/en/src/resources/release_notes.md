# Release Notes

VSAG's official release history and change notes are maintained on GitHub Releases:

- [Releases on GitHub](https://github.com/antgroup/vsag/releases)

Each release includes:

- **Features** — new functionality
- **Improvements**
- **Bug Fixes**
- **Breaking Changes** (when applicable)
- **Contributor credits**

## Versioning

VSAG follows [Semantic Versioning 2.0](https://semver.org/):

- `MAJOR.MINOR.PATCH`
- `MAJOR` generally comes with incompatible API or serialization changes.
- `MINOR` adds functionality while remaining backward compatible.
- `PATCH` contains only bug fixes and performance improvements.

## Getting a Specific Version

### C++ / source

```bash
git checkout vX.Y.Z
make release
```

### Python

```bash
pip install pyvsag==X.Y.Z
```

### Node.js / TypeScript

```bash
npm install vsag@X.Y.Z
```

## Upgrade Guidance

- Read the **Breaking Changes** section of the corresponding release before upgrading across major
  versions.
- When the serialization format changes, validate deserialization compatibility in a staging
  environment first.
- Roll out gradually in production and use the
  [performance evaluation tool](eval.md) to compare recall and latency.
