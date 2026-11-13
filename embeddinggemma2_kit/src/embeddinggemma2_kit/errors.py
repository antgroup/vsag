"""Exception hierarchy for embeddinggemma2_kit.

All errors raised deliberately by the kit derive from :class:`KitError`, so
callers can catch one type. Backend-specific failures are kept distinct because
the recommended handling differs: a missing optional decoder can be fixed by
installing an extra, while a missing vector backend needs an explicit fallback
decision from the caller.
"""

from __future__ import annotations


class KitError(Exception):
    """Base class for all kit errors."""


class ConfigError(KitError):
    """Invalid configuration supplied by the caller."""


class ModelLoadError(KitError):
    """The checkpoint could not be loaded or failed verification."""


class MediaDecodeError(KitError):
    """A media file could not be read or converted to model input."""


class MissingDependencyError(KitError):
    """An optional dependency required for the requested modality is absent.

    Carries the pip extra that would resolve it, so the message can be
    actionable instead of just naming the module.
    """

    def __init__(self, module: str, extra: str, purpose: str) -> None:
        self.module = module
        self.extra = extra
        self.purpose = purpose
        super().__init__(
            f"'{module}' is required for {purpose}. "
            f"Install it with: pip install 'embeddinggemma2-kit[{extra}]'"
        )


class IndexBackendUnavailable(KitError):
    """The requested vector index backend is not installed.

    pyvsag is intentionally not a hard dependency of this package. When it is
    missing, the kit refuses to silently substitute a different backend, because
    HGraph (approximate, graph-based) and the numpy reference (exact, brute
    force) have different recall semantics. Callers must opt in explicitly.
    """

    def __init__(self, backend: str, install_hint: str) -> None:
        self.backend = backend
        self.install_hint = install_hint
        super().__init__(
            f"Vector index backend '{backend}' is unavailable.\n"
            f"{install_hint}\n"
            "If you understand the tradeoff, pass fallback=True to use the exact "
            "numpy brute-force backend instead (slower, but exact)."
        )


class IndexStateError(KitError):
    """An index operation was attempted in an invalid state.

    For example searching before build/add, or loading a persisted index whose
    dimension does not match the current embedder.
    """


class FinetuneError(KitError):
    """Fine-tuning interface misuse or an unsupported training environment."""