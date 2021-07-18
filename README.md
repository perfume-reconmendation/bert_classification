# bert_classification
pretrained 된 transformer의 bert 모델을 이용한 분류기

## dataset
repository : topic_modeling <br>
https://github.com/perfume-reconmendation/topic_modeling

## experiment

- train_ : 15000
  - train : 13500
  - validation : 1500
- test : 5000

- lr : 2e-5
- eps : 1e-8

### 4-type classification model
| #   | data_size  | max_len | epoch_num | batch_size | train_avg_loss | validation_acc | test_acc |
|-----|------------|---------|-----------|------------|----------------|----------------|----------|
| 1.T | 15000, 0.1 | 512     | 1         | 8          | 0.89           | 0.73           |          |
| 2.T | 15000, 0.1 | 512     | 2         | 8          | 0.56           | 0.75           | 0.77     |
| 3.N | 75000, 0.1 | 512     | 1         | 4          | 0.77           | 0.79           | 0.79     |
| 4.N | 75000, 0.1 | 512     | 2         | 4          | 0.57           | 0.81           | 0.81     |
| 5.F | 75000, 0.1 | 256     | 1         | 8          | 0.61           | 0.81           | 0.81     |
| 6.F | 75000, 0.1 | 256     | 2         | 8          | 0.41           | 0.83           | 0.83     |
| 7.F | 75000, 0.1 | 256     | 3         | 8          | 0.28           | 0.82           | 0.82     |


### 89-label classification model
| # | data_size  | epoch_num | batch_size | train_avg_loss | validation_acc | test_acc |
|---|------------|-----------|------------|----------------|----------------|----------|
| 1 | 15000, 0.1 | 1         | 4          | 4.34           | 0.03           | 0.03     |

