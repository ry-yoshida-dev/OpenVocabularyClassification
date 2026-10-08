# result

## Overview

Model-independent classification output shared by every classifier. A `ClassificationResult` keeps the logit of every
prompt query of one image; each class takes its best query, and the `ScoreActivation` of the backend turns the class
logits into scores (softmax over classes for CLIP, independent sigmoids for SigLIP). Its arrays are read-only.

## Components

| Component | Description |
| --------- | ----------- |
| [classification_result.py](./classification_result.py) | `ClassificationResult`: query logits, class logits, matched queries and scores of one image; top class, top-k, threshold filtering and lookup by name. |
| [classification.py](./classification.py) | `Classification`: one class (id, name, score, logit, matched query), yielded by iterating a result. |

## Examples

```python
result = classifier.classify(image, Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",)}))

print(result.top.class_name, result.top.score, result.top.matched_query)
for classification in result.top_k(3):
    print(classification.class_name, classification.score)

present_labels = result.filter_by_score(0.5)
result.score_of("dog")
```
