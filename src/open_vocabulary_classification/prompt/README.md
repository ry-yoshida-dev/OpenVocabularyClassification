# prompt

## Overview

What to classify into. A `Prompt` maps each class name (class id = key order) to the queries the model scores for
it. A class may be queried by several text phrases, several visual queries, or both at once, e.g. `"dog"` by `"dog"`,
`"puppy"` and photos of the user's own dog. The result reports the query of the best embedding of each class
(`Classification.matched_query`).

Text queries are filled into `TextTemplates` (e.g. `"a photo of a {}."`); the templates come from
`ClassifierSettings.text_templates`, so presets carry the templates their model works best with. Each class becomes a
table of queries (rows) and templates (columns); a visual query, which averages the embeddings of its reference images
(few-shot prototypes), fills one cell. An `EmbeddingLayout` lists the embeddings a prompt is scored by, after the
`embedding` steps of the aggregation merged cells into representative embeddings; `EntryGroups` forms the groups of
one step from the `EntryLabel`s of the entries, for embeddings here and for logits in the result.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `Prompt`: class names, queries in query-id order and the class of each query, with validation. |
| [kind.py](./kind.py) | `PromptKind`: `TEXT` or `VISUAL` query. |
| [text_templates.py](./text_templates.py) | `TextTemplates`: validated `{}` templates filled with each text query. |
| [embedding_layout.py](./embedding_layout.py) | `EmbeddingLayout`: the embeddings of a prompt, ordered by class, after the `embedding` aggregation steps. |
| [embedding_entry.py](./embedding_entry.py) | `EmbeddingEntry`: one embedding or a merged group; its class, query and template labels and its sentence and visual members. |
| [entry_label.py](./entry_label.py) | `EntryLabel`: class, query and template an entry's members share, merged along the axis of an aggregation unit. |
| [entry_groups.py](./entry_groups.py) | `EntryGroups`: the groups one aggregation step forms from entry labels, numbered by first appearance, their merged labels and members. |
| [visual_reference.py](./visual_reference.py) | `VisualReference`: reference image, compared by identity. |
| [queries/](./queries/README.md) | Query types: `TextQuery`, `VisualQuery` and their alias `PromptQuery`. |

## Examples

```python
Prompt.from_class_names(("cat", "dog"))
Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat", "kitten")})

my_dog = VisualQuery((VisualReference(photo_1), VisualReference(photo_2)))
Prompt({"my dog": (my_dog,), "other dog": (TextQuery("dog"),)})

TextTemplates(("a photo of a {}.", "a blurry photo of a {}.")).fill("dog")
EmbeddingLayout.of(prompt, templates, PromptAggregation()).entries
```
