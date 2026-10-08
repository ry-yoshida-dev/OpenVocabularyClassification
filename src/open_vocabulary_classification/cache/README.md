# cache

## Overview

Query embeddings cached at the granularity they depend on. A text embedding depends only on its text query (with the
fixed templates of the classifier) and a visual embedding only on its reference image, so each is computed once;
`QueryEmbeddingStore` assembles the per-query embeddings of a prompt from these caches. Text embeddings are kept in a
`LeastRecentlyUsedMapping` bounded by `text_cache_capacity`, so memory stays bounded when text queries keep changing;
reference embeddings are kept in a `WeakKeyDictionary` so they are dropped together with their reference. Every
lookup holds a lock, so threads sharing one classifier keep the caches consistent and embed each query once.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `EmbeddingCache`: thread-safe; computes missing keys in one call, stores values in a given mapping and returns them in key order. |
| [least_recently_used.py](./least_recently_used.py) | `LeastRecentlyUsedMapping`: mapping bounded to a capacity, evicting the least recently read or written entry. |
| [store.py](./store.py) | `QueryEmbeddingStore`: normalized per-query embeddings of a prompt in query-id order, or of bare text queries; ensembles the templates of each text query and averages the references of each visual query. |

## Examples

```python
store = QueryEmbeddingStore(
    embed_sentences=embed_sentences,
    embed_references=embed_references,
    text_templates=TextTemplates(("a photo of a {}.",)),
    text_cache_capacity=4096,
)
query_embeddings = store.embed(prompt)
text_embeddings = store.embed_texts(["dog", "a red car"])
```
