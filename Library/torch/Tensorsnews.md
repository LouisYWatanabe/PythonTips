# Tensorsの新規作成

### 書式&チートシート

```python
import torch

x = torch.empty(2, 3)    # 2行3列の未初期化のテンソルを生成
print(x.shape)
print(x)
print()

x = torch.rand(2, 3)     # 2行3列のランダムに初期化されたテンソルを生成
print(x)
print()

x = torch.zeros(2, 3, dtype=torch.float) # 2行3列の0で初期化しtorch.float型テンソルを生成
print(x.shape)
print(x)
print()

x = torch.ones(2, 3, dtype=torch.float)  # 2行3列の1で初期化し、torch.float型テンソルを生成
print(x.shape)
print(x)
print()

x = torch.tensor([[0.0, 0.1, 0.2],
                  [1.0, 1.1, 1.2]])      # 1行2列のテンソルをリスト値から作成
print()

# 既存のテンソルを使った新規作成
y = x.new_ones(2, 3)   # 2行3列の1で初期化し、既存のテンソルと同じデータ型のテンソルを生成

# 「*_like()」パターン # 既存のテンソルと同じサイズのテンソル（1で初期化、torch.int型）を生成
y = torch.ones_like(x, dtype=torch.int) 
print(y.shape)
print(y)
```

### 対応表

| 宣言方法                                | 処理内容                       |
|-----------------------------------------|--------------------------------|
| torch.empty(行, 列)                 | 指定した行列で未初期化のテンソルを生成      |
| torch.rand(行, 列)                  | 指定した行列でランダムに初期化されたテンソルを生成       |
| torch.zeros(行, 列, dtype=torch.float)              | 指定した行列の0で初期化しtorch.float型テンソルを生成       |
| torch.ones(行, 列, dtype=torch.float)              | 指定した行列の1で初期化しtorch.float型テンソルを生成       |
| torch.tensor([ [行, 列], [行, 列] ])                  | 指定した行列ののテンソルをリスト値から作成      |
| torch.ones_like(＜既存のテンソル＞, dtype=torch.int)                  | 既存のテンソルと同じサイズのテンソル（1で初期化、torch.int型）を生成      |


### 例

```python
import numpy as np
import torch

x = torch.empty(2, 3)    # 2行3列の未初期化のテンソルを生成
print(x.shape)
print(x)
print()

x = torch.rand(2, 3)     # 2行3列のランダムに初期化されたテンソルを生成
print(x)
print()

x = torch.zeros(2, 3, dtype=torch.float) # 2行3列の0で初期化しtorch.float型テンソルを生成
print(x.shape)
print(x)
print()

x = torch.ones(2, 3, dtype=torch.float)  # 2行3列の1で初期化し、torch.float型テンソルを生成
print(x.shape)
print(x)
print()

x = torch.tensor([[0.0, 0.1, 0.2],
                  [1.0, 1.1, 1.2]])      # 2行3列のテンソルをリスト値から作成
print(x)
print()

# 既存のテンソルを使った新規作成
y = x.new_ones(2, 3)   # 2行3列の1で初期化し、既存のテンソルと同じデータ型のテンソルを生成

# 「*_like()」パターン # 既存のテンソルと同じサイズのテンソル（1で初期化、torch.int型）を生成
y = torch.ones_like(x, dtype=torch.int) 
print(y.shape)
print(y)
```

	torch.Size([2, 3])
	tensor([[0., 0., 0.],
			[0., 0., 0.]])

	tensor([[0.3649, 0.8285, 0.7699],
			[0.6017, 0.4490, 0.1506]])

	torch.Size([2, 3])
	tensor([[0., 0., 0.],
			[0., 0., 0.]])

	torch.Size([2, 3])
	tensor([[1., 1., 1.],
			[1., 1., 1.]])


	tensor([[0.0000, 0.1000, 0.2000],
        	[1.0000, 1.1000, 1.2000]])

	torch.Size([2, 3])
	tensor([[1, 1, 1],
			[1, 1, 1]], dtype=torch.int32)

### 説明
