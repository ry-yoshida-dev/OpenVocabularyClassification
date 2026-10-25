# cache

## Overview

Prompt embeddings cached at the granularity they depend on. A sentence embedding depends only on its sentence (a
text query filled into one template) and a reference embedding only on its reference image, so each is computed
once; `QueryEmbeddingStore` assembles the embeddings of an `EmbeddingLayout` from these caches, whichever templates
and template embedding the layout uses. Sentence embeddings are kept in a `LeastRecentlyUsedMapping` bounded by
`sentence_cache_capacity`, so memory stays bounded when text queries or templates keep changing (every cached
embedding owns its storage, so eviction frees it);
reference embeddings are kept in a `WeakKeyDictionary` so they are dropped together with their reference. Every
lookup holds a lock, so threads sharing one classifier keep the caches consistent and embed each query once.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `EmbeddingCache`: thread-safe; computes missing keys in one call, stores values in a given mapping and returns them in key order. |
| [least_recently_used.py](./least_recently_used.py) | `LeastRecentlyUsedMapping`: mapping bounded to a capacity, evicting the least recently read or written entry. |
| [store.py](./store.py) | `QueryEmbeddingStore`: normalized embeddings of a prompt layout in embedding-id order, or of bare text queries ensembled over templates; averages the sentences of each entry and the references of each visual query. |

## Examples

```python
store = QueryEmbeddingStore(
    embed_sentences=embed_sentences,
    embed_references=embed_references,
    sentence_cache_capacity=16384,
)
templates = TextTemplates(("a photo of a {}.",))
prompt_embeddings = store.embed(EmbeddingLayout.of(prompt, templates, TemplateCombination.SEPARATE))
text_embeddings = store.embed_texts(["dog", "a red car"], templates)
```
