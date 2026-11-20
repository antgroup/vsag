"""Role-to-prompt mapping.

EmbeddingGemma 2 is an asymmetric retrieval model: queries and documents are
prefixed differently, and using the wrong prefix does not raise -- it quietly
loses retrieval quality. Measured on the Flickr30k baseline used by this repo
(768-d, bf16), text->image hit@1 is 0.8632 when the correct prefixes are used.
A prefix swap does not error; it just returns worse neighbours.

To make that failure mode unreachable, this kit exposes an explicit ``role``
argument and never lets callers pass raw prompt names.
"""

from __future__ import annotations

from ..constants import PROMPT_DOCUMENT, PROMPT_QUERY
from ..errors import ConfigError

# Public role vocabulary. "query" is the short thing you search WITH;
# "document" is the thing being searched FOR.
ROLE_QUERY = "query"
ROLE_DOCUMENT = "document"

_ROLE_TO_PROMPT = {
    ROLE_QUERY: PROMPT_QUERY,
    ROLE_DOCUMENT: PROMPT_DOCUMENT,
}

VALID_ROLES = tuple(_ROLE_TO_PROMPT)


def prompt_for_role(role: str) -> str:
    """Map a public role onto the checkpoint's prompt name.

    Raises:
        ConfigError: if ``role`` is not one of :data:`VALID_ROLES`.
    """
    try:
        return _ROLE_TO_PROMPT[role]
    except KeyError:
        raise ConfigError(
            f"unknown role {role!r}; expected one of {list(VALID_ROLES)}. "
            f"'{ROLE_QUERY}' is used for the search query and "
            f"'{ROLE_DOCUMENT}' for corpus items."
        ) from None