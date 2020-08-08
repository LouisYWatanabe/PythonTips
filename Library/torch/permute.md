# PyTorchテンソル軸（データ格納）の順番を変更

### 書式&チートシート

```python
import torch
import numpy as np

# 軸変更前のサイズの確認
print(y_seq_t_train.size())

# データの順番を変更
# LSTM層で扱えるように変更
y_seq_t_train = y_seq_t_train.permute(1, 0)
y_target_t_train = y_target_t_train.permute(1, 0)

# 軸変更後のサイズの確認
print(y_seq_t_train.size())
```
	torch.Size([450, 40])
	torch.Size([40, 450])