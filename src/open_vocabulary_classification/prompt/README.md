# prompt

## Overview

What to classify into. A `Prompt` maps each class name (class id = key order) to the queries the model scores for
it. A class may be queried by several text phrases, several visual queries, or both at once, e.g. `"dog"` by `"dog"`,
`"puppy"` and photos of the user's own dog. The result reports the highest query of each class
(`Classification.matched_query`).

Text queries are filled into `TextTemplates` (e.g. `"a photo of a {}."`); the templates come from
`ClassifierSettings.text_templates`, so presets carry the templates their model works best with. An `EmbeddingLayout`
lists the embeddings a prompt is scored by: one per visual query, which averages the embeddings of its reference
images (few-shot prototypes), and per text query one ensembling its sentences (`TemplateCombination.ENSEMBLE`) or one
per sentence (any other `TemplateCombination`).

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `Prompt`: class names, queries in query-id order and the class of each query, with validation. |
| [kind.py](./kind.py) | `PromptKind`: `TEXT` or `VISUAL` query. |
| [text_templates.py](./text_templates.py) | `TextTemplates`: validated `{}` templates filled with each text query. |
| [embedding_layout.py](./embedding_layout.py) | `EmbeddingLayout`: the embeddings of a prompt in embedding-id order and the embeddings of each query. |
| [embedding_entry.py](./embedding_entry.py) | `EmbeddingEntry`: one embedding, its query, the sentences averaged into it and its template. |
| [visual_reference.py](./visual_reference.py) | `VisualReference`: reference image, compared by identity. |
| [queries/](./queries/README.md) | Query types: `TextQuery`, `VisualQuery` and their alias `PromptQuery`. |

## Examples

```python
Prompt.from_class_names(("cat", "dog"))
Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat", "kitten")})

my_dog = VisualQuery((VisualReference(photo_1), VisualReference(photo_2)))
Prompt({"my dog": (my_dog,), "other dog": (TextQuery("dog"),)})

TextTemplates(("a photo of a {}.", "a blurry photo of a {}.")).fill("dog")
EmbeddingLayout.of(prompt, templates, TemplateCombination.SEPARATE).entry_query_ids
```
