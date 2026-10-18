"""Package-wide constants and small helpers.

The values here mirror the verified EmbeddingGemma 2 checkpoint that the
earlier CPU evaluation experiments used, so the kit can assert it loaded the
same model rather than silently drifting to another revision.
"""

from __future__ import annotations

# Output embedding dimension of the full multimodal model. The text tower has a
# 512-wide hidden state and projects to 768; vision/audio towers project into
# the same 768-wide space, which is what makes cross-modal search possible.
EMBEDDING_DIM = 768

# Exact parameter count of the verified 744M checkpoint. Used as a load-time
# assertion so a wrong/partial checkpoint fails loudly instead of producing
# plausible-looking garbage embeddings.
EXPECTED_PARAMETER_COUNT = 744_371_488

# Revision of the checkpoint used by the baseline experiments. Recorded in
# index metadata for reproducibility.
BASELINE_MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"

# Audio is resampled to this rate; the audio tower expects 16 kHz mono.
AUDIO_SAMPLE_RATE = 16_000

# Prompt names shipped in config_sentence_transformers.json. Queries and
# documents MUST use different prefixes -- mixing them degrades retrieval
# silently rather than raising, so callers never pass these directly.
PROMPT_QUERY = "SearchQuery"
PROMPT_DOCUMENT = "Document"

# Video preprocessing accepted by the verified encoder: caller-supplied frames
# are used as-is (no internal sampling) and timestamps are added.
VIDEO_PROCESSING_KWARGS = {
    "video": {"do_sample_frames": False, "add_timestamps": True, "max_soft_tokens": 70}
}

# Modality keys accepted by the mixed-modality ``embed`` entry point.
SUPPORTED_MODALITIES = ("text", "image", "audio", "video")