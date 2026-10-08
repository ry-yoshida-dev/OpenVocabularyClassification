# queries

## Overview

Query types telling the model what a prompt class looks like. A class may hold any mix of them.

## Components

| Component | Description |
| --------- | ----------- |
| [text.py](./text.py) | `TextQuery`: a stripped, non-blank phrase, filled into the text templates; `kind` is `TEXT`. |
| [visual.py](./visual.py) | `VisualQuery`: reference images averaged into one query; `kind` is `VISUAL`. |
| [core.py](./core.py) | `PromptQuery`: alias for `TextQuery \| VisualQuery`. |

## Examples

```python
TextQuery("puppy")
VisualQuery((VisualReference(mug_image_1), VisualReference(mug_image_2)))
```
