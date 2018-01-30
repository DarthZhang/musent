# musent
Used for several NLP tasks, musent is a collection of small-footprint models. For example, at inference time for the SST binary sentiment classification task, the micro-model performs no more than a few thousand multiplies per word vector to achieve comparable accuracy, with respect to previous state-of-the-art. Even more drastically, the nano-model uses only a few dozen multiplies per **sentence** to achieve interesting results.

## Installation

1. Run `./getData.sh`
2. Run `python microsem` to train a model (SST binary by default)

## Results
| Model          | SST-fine      | SST-bin    | # mult/word  |
| -------------- |---------------|------------|--------------|
| CNN-static     | 45.5          | 86.8       | ~90k         |
| CNN-multi      | 47.4          | 88.1       | ~180k        |
| CNN-nonstatic  | 48.0          | 87.2       | ~90k         |
| musent-micro   | 47.7          | 85.9       | ~1.6k        |
| musent-nano    | 40.6          | 83.4       | < 100        |

WIP
