# result

## Overview

Model-independent classification output shared by every classifier. A `ClassificationResult` keeps the logit of every
prompt query of one image; each class takes its best query, and the `ScoreActivation` of the backend turns the class
logits into scores (softmax over classes for CLIP, independent sigmoids for SigLIP). Its arrays are read-only.

A softmax score is relative to the other classes of the prompt, so CLIP always gives some class most of the score,
even when the image shows none of them. To reject such images, add a background class (e.g. `"something else"`) or
threshold the logit with `filter_by_logit`, which does not depend on the other classes; choose the threshold on real
data. `to_records` converts a result to flat, JSON-serializable `ClassificationRecord` dictionaries.

## Components

| Component | Description |
| --------- | ----------- |
| [classification_result.py](./classification_result.py) | `ClassificationResult`: query logits, class logits, matched queries and scores of one image; top class, top-k, score and logit thresholds, lookup by name and records. |
| [classification.py](./classification.py) | `Classification`: one class (id, name, score, logit, matched query), yielded by iterating a result. |
| [classification_record.py](./classification_record.py) | `ClassificationRecord`: flat, JSON-serializable form of a `Classification`. |

## Examples

```python
result = classifier.classify(image, Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",)}))

print(result.top.class_name, result.top.score, result.top.matched_query)
for classification in result.top_k(3):
    print(classification.class_name, classification.score)

present_labels = result.filter_by_score(0.5)
confident_labels = result.filter_by_logit(25.0)
result.score_of("dog")
json.dumps(result.to_records())
```
