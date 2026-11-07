# prompt

## Overview

What to classify into. A `Prompt` maps each class name to the queries the model scores for it: several text phrases, visual queries, or both, e.g. `"dog"` by `"dog"`, `"puppy"` and photos of the user's own dog. Text queries are filled into the `text_templates` of the settings, and the embeddings of a class are laid out for aggregation.

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `Prompt`, the classes and their queries. |
| [kind.py](./kind.py) | `PromptKind`: `TEXT` or `VISUAL` query. |
| [text_templates.py](./text_templates.py) | `TextTemplates`, `{}` templates filled with each text query. |
| [embedding_layout.py](./embedding_layout.py) | `EmbeddingLayout`, the embeddings a prompt is scored by. |
| [embedding_entry.py](./embedding_entry.py) | `EmbeddingEntry`, one embedding or a merged group. |
| [entry_label.py](./entry_label.py) | `EntryLabel`, the class, query and template of an entry. |
| [entry_groups.py](./entry_groups.py) | `EntryGroups`, the groups one aggregation step forms. |
| [visual_reference.py](./visual_reference.py) | `VisualReference`, a reference image. |
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
