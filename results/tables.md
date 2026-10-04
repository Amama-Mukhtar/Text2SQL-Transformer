## Table 1 - Data (lengths include <s> and </s>)

| | Train | Dev | Test |
|---|---|---|---|
| Pairs | 56355 | 8421 | 15878 |
| Mean / max source length | 42.5 / 222 | 42.5 / 167 | 42.7 / 260 |
| Mean / max target length | 14.8 / 65 | 14.8 / 44 | 14.9 / 46 |
| Pairs dropped as too long | 19 | 0 | 0 |

## Table 2 - Model and training

- Trainable parameters: 7,577,600
- Epochs trained / best epoch: 20 / 18
- Best dev loss: 1.4455
- Training time and GPU: 19.8 min on Tesla T4

## Table 3 - Official metrics

| Split | Decoding | Logical form (%) | Execution (%) | Parse failures (%) |
|---|---|---|---|---|
| Dev | greedy | 65.29 | 71.56 | 0.0 |
| Dev | beam | 65.36 | 71.61 | 0.01 |
| Test | beam | 65.05 | 71.39 | 0.09 |

## Table 4 - Component accuracy (dev, greedy)

- sel column correct: 93.4%
- agg correct: 89.76%
- WHERE clause correct: 75.19%

## Table 4 - Component accuracy (dev, beam)

- sel column correct: 93.47%
- agg correct: 89.75%
- WHERE clause correct: 75.37%

## Gold round-trip (dev gold targets -> our parser -> official evaluator)

- Logical form 100.0%, execution 100.0%, parse failures 0.0%
