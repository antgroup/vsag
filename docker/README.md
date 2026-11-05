# CI image tags

Use `<repository>:ci-<line>-<variant>` when selecting a hosted CI image.
The main branch publishes the `main` line. Each maintenance branch publishes
its major and minor version with a leading `v`, for example `v1.0` or `v1.1`.
Patch releases share their maintenance line. The workflow publishes from pushes
to its own branch when the image inputs change. Pull requests build and check
the images on native runners.

| Example tag | Platforms |
| --- | --- |
| `vsaglib/vsag:ci-main-base` | linux/amd64, linux/arm64 |
| `vsaglib/vsag:ci-main-openblas` | linux/amd64, linux/arm64 |
| `vsaglib/vsag:ci-v1.0-openblas` | linux/amd64, linux/arm64 |
| `vsaglib/vsag:ci-v1.1-mkl` | linux/amd64 |

Docker selects the platform for the current engine. Use
`docker buildx imagetools inspect vsaglib/vsag:ci-main-openblas` to inspect
the published platforms. The same tag scheme applies in GHCR for lines that
publish there. Hosted workflows use the line tag. Team CI selects its reviewed
image by digest through the existing admission process.

The publisher builds each backend against the base from the same native build.
Run-specific staging tags carry the platform images to the registry. A final
job assembles the public tags after both native builds succeed. Switch a
workflow to a new tag after confirming that its registry lists the required
platforms.
