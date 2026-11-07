# cache

## Overview

Thread-safe prompt embeddings cached per sentence (a text query filled into one template) and per reference image, so prompts that share them do not recompute them.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `EmbeddingCache`, thread-safe embeddings computed per key and kept for later prompts. |
| [least_recently_used.py](./least_recently_used.py) | `LeastRecentlyUsedMapping`, a mapping bounded to a capacity. |
| [store.py](./store.py) | `QueryEmbeddingStore`, which assembles prompt and text embeddings from the caches. |

## Examples

```python
store = QueryEmbeddingStore(
    embed_sentences=embed_sentences,
    embed_references=embed_references,
    sentence_cache_capacity=16384,
)
templates = TextTemplates(("a photo of a {}.",))
prompt_embeddings = store.embed(EmbeddingLayout.of(prompt, templates, PromptAggregation()))
text_embeddings = store.embed_texts(["dog", "a red car"], templates)
```
