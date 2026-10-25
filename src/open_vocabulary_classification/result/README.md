# result

## Overview

Model-independent classification output shared by every classifier. A `ClassificationResult` keeps the logit of every
prompt embedding of one image and combines them per class in two steps: the template combination turns the sentence
logits of each text query into one (or, with `separate`, keeps every sentence as a candidate), and the query combination
turns the candidates into the class logit. The `ScoreActivation` of the backend then turns the class logits into
scores (softmax over classes for CLIP, independent sigmoids for SigLIP). Its arrays are read-only. A `LogitAggregation`
is prepared once per prompt; since embeddings are grouped by query and queries by class, both steps are vectorized
`LogitSegments` reductions over consecutive logits, so scoring an image takes a few NumPy calls.

A softmax score is relative to the other classes of the prompt, so CLIP always gives some class most of the score,
even when the image shows none of them. To reject such images, add a background class (e.g. `"something else"`) or
threshold the logit with `filter_by_logit`, which does not depend on the other classes; choose the threshold on real
data, keeping in mind that a `max` over more queries or sentences per class is biased upward. `to_records` converts a result to flat, JSON-serializable `ClassificationRecord` dictionaries.

## Components

| Component | Description |
| --------- | ----------- |
| [classification_result.py](./classification_result.py) | `ClassificationResult`: embedding logits, class logits, matched queries and templates and scores of one image; top class, top-k, score and logit thresholds, lookup by name and records. |
| [logit_aggregation.py](./logit_aggregation.py) | `LogitAggregation`: layout and combinations of a prompt, combining the embedding logits of one image into class logits. |
| [logit_segments.py](./logit_segments.py) | `LogitSegments`: consecutive runs of a logit vector, combined (`max`, `mean`, `median`, `min`) and arg-maxed run by run. |
| [class_logits.py](./class_logits.py) | `ClassLogits`: class logits of one image and the embedding each class matched best. |
| [classification.py](./classification.py) | `Classification`: one class (id, name, score, logit, matched query and template), yielded by iterating a result. |
| [classification_record.py](./classification_record.py) | `ClassificationRecord`: flat, JSON-serializable form of a `Classification`. |

## Examples

```python
result = classifier.classify(image, Prompt.from_texts({"dog": ("dog", "puppy"), "cat": ("cat",)}))

print(result.top.class_name, result.top.score, result.top.matched_query, result.top.matched_template)
for classification in result.top_k(3):
    print(classification.class_name, classification.score)

present_labels = result.filter_by_score(0.5)
confident_labels = result.filter_by_logit(25.0)
result.score_of("dog")
json.dumps(result.to_records())
```
