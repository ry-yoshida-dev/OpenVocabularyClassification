# result

## Overview

Model-independent classification output shared by every classifier: the logits of every prompt embedding of one image, reduced to class scores by the aggregation and the score activation of the backend. Logits stored with their labels can be scored again with other aggregation steps without the prompt or the model.

## Components

| Component | Description |
| --------- | ----------- |
| [classification_result.py](./classification_result.py) | `ClassificationResult`, the class scores of one image, with ranking, thresholds and records. |
| [logit_aggregation.py](./logit_aggregation.py) | `LogitAggregation`, which reduces the embedding logits of one image to candidate logits. |
| [class_scores.py](./class_scores.py) | `ClassScores`, the score, logit and matched embedding of every class. |
| [logit_reduction.py](./logit_reduction.py) | `LogitReduction`, one `similarity` step. |
| [logit_segments.py](./logit_segments.py) | `LogitSegments`, consecutive runs of a logit vector reduced run by run. |
| [candidate_logits.py](./candidate_logits.py) | `CandidateLogits`, the logits of the candidates left for one image. |
| [classification.py](./classification.py) | `Classification`, one class, yielded by iterating a result. |
| [classification_record.py](./classification_record.py) | `ClassificationRecord`, the JSON-serializable form of a `Classification`. |

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

labels = result.layout.entry_labels
median = PromptAggregation.of(
    [AggregationStep(AggregationUnit.PROMPT, AggregationTiming.SIMILARITY, AggregationMethod.MEDIAN)]
)
ClassScores.of(LogitAggregation(labels, median), result.embedding_logits, ScoreActivation.SOFTMAX).scores
```
