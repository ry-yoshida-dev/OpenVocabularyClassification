# prompt

## Overview

What to classify into. A `Prompt` maps each class name (class id = key order) to the queries the model scores for
it. A class may be queried by several text phrases, several visual queries, or both at once, e.g. `"dog"` by `"dog"`,
`"puppy"` and photos of the user's own dog. Each class takes the logit of its best query, and the result reports that
query (`Classification.matched_query`).

Text queries are filled into `TextTemplates` (e.g. `"a photo of a {}."`) and the filled sentences are ensembled into
one embedding; the templates come from `ClassifierSettings.text_templates`, so presets carry the templates their
model works best with. Visual queries average the embeddings of their reference images (few-shot prototypes).

## Components

| Component | Description |
| --------- | ----------- |
| [core.py](./core.py) | `Prompt`: class names, queries in query-id order and the class of each query, with validation. |
| [kind.py](./kind.py) | `PromptKind`: `TEXT` or `VISUAL` query. |
| [text_templates.py](./text_templates.py) | `TextTemplates`: validated `{}` templates filled with each text query for prompt ensembling. |
| [visual_reference.py](./visual_reference.py) | `VisualReference`: reference image, compared by identity. |
| [queries/](./queries/README.md) | Query types: `TextQuery`, `VisualQuery` and their alias `PromptQuery`. |

## Examples

```python
Prompt.from_class_names(("cat", "dog"))
Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat", "kitten")})

my_dog = VisualQuery((VisualReference(photo_1), VisualReference(photo_2)))
Prompt({"my dog": (my_dog,), "other dog": (TextQuery("dog"),)})

TextTemplates(("a photo of a {}.", "a blurry photo of a {}.")).fill("dog")
```
