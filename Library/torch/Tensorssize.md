# Tensorsのサイズ取得とサイズ変更

### 書式&チートシート

```python
import torch

# テンソルの新規作成
x = torch.empty(2, 3)
x = torch.rand_like(x, dtype=torch.float)

print(x.size())        # thorchの行列サイズを出力
print(x.shape)         # thorchの行列サイズを出力

print()
print(len(x))          # 行数（データ数の取得）
print(x.ndim)          # 列数（テンソルの次元数）の取得
print()

# テンソルサイズ変更
z = x.view(3, 2)       # 3行2列にサイズを変更
print(z.shape)         # 変更したサイズの取得
```

### 対応表

| 宣言方法                                | 処理内容                       |
|-----------------------------------------|--------------------------------|
| ＜モジュール名＞.size()                 | Thorchの行列サイズを出力       |
| ＜モジュール名＞.shape                  | Thorchの行列サイズを出力       |
| len(＜モジュール名＞)                   | 行数の取得                     |
| ＜モジュール名＞.ndim                   | 列数（テンソルの次元数）の取得 |
| ＜モジュール名＞.view(変更したい行, 列) | 3行2列にサイズを変更           |

### 例

```python
import torch

# テンソルの新規作成
x = torch.empty(2, 3)
x = torch.rand_like(x, dtype=torch.float)

print(x.size())        # thorchの行列サイズを出力
print(x.shape)         # thorchの行列サイズを出力

print()
print(len(x))          # 行数（データ数の取得）
print(x.ndim)          # 列数（テンソルの次元数）の取得
print()

# テンソルサイズ変更
z = x.view(3, 2)       # 3行2列にサイズを変更
print(z.shape)         # 変更したサイズの取得
```

	torch.Size([2, 3])
	torch.Size([2, 3])

	2
	2

	torch.Size([3, 2])

### 説明
