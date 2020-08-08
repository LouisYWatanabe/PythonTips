# Tensor の作成

多次元配列のこと

### 書式

```python
# 初期化されていない5x3行列を作成
x = torch.empty(5, 3)
# ランダムに初期化された行列を作成
x = torch.rand(5, 3)
# 要素が0の Tensor を作成
x = torch.zeros(5, 3, dtype=torch.long)
# 要素が1の Tensor を作成
x = x.new_ones(5, 3, dtype=torch.double)
x = torch.ones(5, 3, dtype=torch.double)
# リストから直接Tensorを作成
x = torch.tensor([5.5, 3])
# float型のtensorを作成する
x = torch.FloatTensor([10])
# 標準化（平均0、標準偏差1）のランダム値の Tensor を作成
x = torch.rand_like(x, dtype=torch.float)
```

### 引数

### 例



```python
from __future__ import print_function
import torch

print(torch.__version__)
```

    1.5.0+cpu
    

Tensor とは任意の次元の配列です。 数字1つやスカラーは0次テンソル、1次元配列やベクトルは1次テンソル、2次元配列や行列は2次テンソル、3次元配列は3次テンソルになります。 torch.empty で初期化されていない Tensor を作成します。

初期化されていない5x3行列を作成します。


```python
x = torch.empty(5, 3)
print(x)
```

    tensor([[9.0000e-39, 1.0561e-38, 1.0653e-38],
            [4.1327e-39, 8.9082e-39, 9.8265e-39],
            [9.4592e-39, 1.0561e-38, 1.0653e-38],
            [1.0469e-38, 9.5510e-39, 1.0745e-38],
            [9.6429e-39, 1.0561e-38, 9.1837e-39]])
    

ランダムに初期化された行列を作成します。


```python
x = torch.rand(5, 3)
x
```




    tensor([[0.8739, 0.0151, 0.6881],
            [0.4690, 0.7814, 0.9475],
            [0.7042, 0.6132, 0.5430],
            [0.2911, 0.4520, 0.1730],
            [0.2077, 0.0417, 0.5644]])



要素が0の Tensor を作成


```python
x = torch.zeros(5, 3, dtype=torch.long)
print(x)
```

    tensor([[0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]])
    

要素が1の Tensor を作成


```python
x = torch.ones(5, 3, dtype=torch.double)
print(x)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)
    


```python
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
```

    tensor([[1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.],
            [1., 1., 1.]], dtype=torch.float64)
    

リストから直接Tensorを作成


```python
x = torch.tensor([5.5, 3])
print(x)
```

    tensor([5.5000, 3.0000])
    

標準化（平均0、標準偏差1）のランダム値の Tensor を作成


```python
x = torch.rand_like(x, dtype=torch.float)
print(x)
```

    tensor([0.3699, 0.9856])
    
```python
print(x.size())
```
    torch.Size([5, 3])

### 説明

PyTorch では入力データを Tensor で扱います。
Tensor とは任意の次元の配列です。
数字1つやスカラーは0次テンソル、1次元配列やベクトルは1次テンソル、2次元配列や行列は2次テンソル、3次元配列は3次テンソルになります。
torch.empty で初期化されていない Tensor を作成します。