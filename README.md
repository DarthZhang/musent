# microsem
Used for several NLP tasks, microsem is designed to be extremely small-footprint. For example, at inference time for the SST binary sentiment classification task, the model performs no more than a few dozen multiplies to achieve decent accuracy, compared to baseline.

## Installation

1. Run `./getData.sh`
2. Run `python microsem` to train a model (SST binary by default)

## Best Results
| Task          | Best dev      | Test       | # multiplies |
| ------------- |---------------|------------|--------------|
| SST fine      | 43.7          | 40.6       | 80           |
| SST binary    | 83.8          | 83.4       | 32           |

WIP
