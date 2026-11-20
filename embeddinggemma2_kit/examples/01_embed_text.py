"""Encode text as query and as document; compare the two spaces."""

from embeddinggemma2_kit import Embedder

emb = Embedder(dtype="bfloat16")

passage = "The man with pierced ears is wearing glasses and an orange hat."

# The same string encoded with different roles lands in different places,
# because the task prefix differs. That asymmetry is what retrieval relies on.
as_query = emb.embed_text([passage], role="query")
as_doc = emb.embed_text([passage], role="document")

print("query   :", as_query.shape, "norm", float((as_query ** 2).sum() ** 0.5))
print("document:", as_doc.shape, "norm", float((as_doc ** 2).sum() ** 0.5))
print("cosine between the two roles:", float((as_query * as_doc).sum()))

# A query should match a semantically equal document better than a different one.
others = emb.embed_text(
    ["A black and white dog runs through a grassy garden."], role="document"
)
print("similarity to same content:", float((as_query * as_doc).sum()))
print("similarity to other content:", float((as_query * others).sum()))
