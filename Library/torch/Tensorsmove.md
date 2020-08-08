# Tensor の操作

### 書式

```python
# Tensor同士の加算
x+y
torch.add(x, y)
y.add_(x)

# スライス操作
x[:, 1]

# Tensorのサイズを変更
x = torch.randn(4, 4)    # 4行4列のテンソル
y = x.view(16)           # 1行16列のテンソル
z = x.view(-1, 8)        # 2行8列のテンソル
# 要素1の値の取得
x = torch.randn(1)
print(x)
print(x.item())

# TensorからNumpyへの変換
a = torch.ones(5)
b = a.numpy()			# TensorからNumpyへの変換
a.add_(1)				# メモリを共有の確認
print(a)
print(b)

# NumpyからTensorへの変換
a = np.ones(5)
b = torch.from_numpy(a) # NumpyからTensorへの変換
b.add_(1)
print(a)
print(b)
```

### 引数

### 例

```python
from __future__ import print_function
import torch

print(torch.__version__)
```

    1.5.0+cpu
    


```python
x = torch.empty(5, 3)                      # 初期化されていない5x3行列を作成
x = torch.rand_like(x, dtype=torch.float)  # 標準化（平均0、標準偏差1）のランダム値の Tensor を作成
print(x)
```

    tensor([[0.9607, 0.6301, 0.0680],
            [0.5437, 0.2938, 0.8365],
            [0.5782, 0.2813, 0.5448],
            [0.3236, 0.9968, 0.5137],
            [0.9359, 0.4149, 0.1525]])
    


```python
y = torch.rand(5, 3)
y = torch.rand_like(y, dtype=torch.float)
print(y)
```

    tensor([[0.8504, 0.0130, 0.4950],
            [0.3421, 0.8210, 0.1851],
            [0.4200, 0.8043, 0.9973],
            [0.3947, 0.7930, 0.0714],
            [0.0388, 0.8175, 0.8353]])
    

### 加算

加算は、要素単位で行われます。

`+` 演算子と `torch.add` の2通りの記述があります。


```python
print(x+y)
print()
print(torch.add(x, y))
```

    tensor([[1.8111, 0.6432, 0.5631],
            [0.8859, 1.1148, 1.0216],
            [0.9982, 1.0855, 1.5421],
            [0.7183, 1.7898, 0.5851],
            [0.9747, 1.2324, 0.9879]])
    
    tensor([[1.8111, 0.6432, 0.5631],
            [0.8859, 1.1148, 1.0216],
            [0.9982, 1.0855, 1.5421],
            [0.7183, 1.7898, 0.5851],
            [0.9747, 1.2324, 0.9879]])
    

out 引数で、出力 Tensor を指定することができます。


```python
result = torch.empty(5, 3)          # 出力したいTensor
print(torch.add(x, y, out=result))
print()
print(result)
```

    tensor([[1.8111, 0.6432, 0.5631],
            [0.8859, 1.1148, 1.0216],
            [0.9982, 1.0855, 1.5421],
            [0.7183, 1.7898, 0.5851],
            [0.9747, 1.2324, 0.9879]])
    
    tensor([[1.8111, 0.6432, 0.5631],
            [0.8859, 1.1148, 1.0216],
            [0.9982, 1.0855, 1.5421],
            [0.7183, 1.7898, 0.5851],
            [0.9747, 1.2324, 0.9879]])
    

`add_()`で入力 Tensor を計算結果で書き換えます。


```python
print(torch.add(x, y))
print()
y.add_(x)
print(y)
```

    tensor([[1.8111, 0.6432, 0.5631],
            [0.8859, 1.1148, 1.0216],
            [0.9982, 1.0855, 1.5421],
            [0.7183, 1.7898, 0.5851],
            [0.9747, 1.2324, 0.9879]])
    
    tensor([[1.8111, 0.6432, 0.5631],
            [0.8859, 1.1148, 1.0216],
            [0.9982, 1.0855, 1.5421],
            [0.7183, 1.7898, 0.5851],
            [0.9747, 1.2324, 0.9879]])
    

NumPyのようにスライスを使用することができます


```python
# 1行取得
print(x[:, 1])
```

    tensor([0.6301, 0.2938, 0.2813, 0.9968, 0.4149])
    

### サイズ変更
テンソルのサイズを変更をしたい場合は、`torch.view`を使用できます。

1を指定すると、他の次元を考慮して補完されます。


```python
x = torch.randn(4, 4)    # 4行4列のテンソル
y = x.view(16)           # 1行16列のテンソル
z = x.view(-1, 8)        # 2行8列のテンソル

print(x.size(), y.size(), z.size())
```

    torch.Size([4, 4]) torch.Size([16]) torch.Size([2, 8])
    

要素数1の Tensor に対して item() を利用すると、通常の値として取得できます。


```python
x = torch.randn(1)
print(x)
print(x.item())
```

    tensor([-0.0066])
    -0.006623469293117523
    

### TensorからNumpyへの変換
Tensorから NumPy への変換は、`torch.numpy()` で行います。

メモリを共有するため、一方を変更すると、もう一方も変更されます。


```python
a = torch.ones(5)
b = a.numpy()
print(b)
```

    [1. 1. 1. 1. 1.]
    


```python
a.add_(1)				# bも同時にインクリメントされます
print(a)
print(b)
```

    tensor([2., 2., 2., 2., 2.])
    [2. 2. 2. 2. 2.]
    

### NumpyからTensorへの変換
NumPy から Tensor への変換は、`torch.from_numpy()` で行います。

メモリを共有するため、一方を変更すると、もう一方も変更されます。


```python
import numpy as np

a = np.ones(5)
b = torch.from_numpy(a)
print(a)
print(b)

print()

b.add_(1)				# メモリを共有の確認
print(a)
print(b)
```

    [1. 1. 1. 1. 1.]
    tensor([1., 1., 1., 1., 1.], dtype=torch.float64)
    [2. 2. 2. 2. 2.]
    tensor([2., 2., 2., 2., 2.], dtype=torch.float64)
    

### CUDA Tensor
`torch.to() `を利用して Tensor を様々なデバイスに移動できます。
以下のコードでは CUDA デバイスに移動しています。
CUDA は NVIDIA が提供している、GPU環境のプラットフォームです。


```python
# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!
```

GooglrColaboratoryの出力結果
```
tensor([-0.8088], device='cuda:0')
tensor([-0.8088], dtype=torch.float64)
```




### 説明