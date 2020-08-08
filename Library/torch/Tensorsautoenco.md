# 自動微分

### 書式

```python
x = torch.tensor([2], dtype=torch.float, requires_grad=True)
y = 3 * x**3
# xについて誤差逆伝播を実行
y.backward()
# 勾配計算・微分の結果を求める 3*3*2**2=36
x.grad
```

### 引数

	requires_grad=True Trueと置くことで勾配計算が使用できるようになります。

### 例
自動微分

$y=3x^3$

$\frac{dy}{dx} = 3×3x^2$

$y'=9x^2$

```python
# 2のテンソルを勾配計算ありで作成
x = torch.tensor([2], dtype=torch.float, requires_grad=True)
# 勾配計算の実行式の定義
y = 3 * x**3
# xについて誤差逆伝播を実行
y.backward()
# 微分の結果を求める 3*3*2**2=36
x.grad
```
	tensor([36.])

別な例 複数の変数があるときの勾配計算
```python
x1 = torch.tensor([4], dtype=torch.float, requires_grad=True)
x2 = torch.tensor([2], dtype=torch.float, requires_grad=True)

y = 2 * x1**2 + 3 * x2**3
y.backward()
# 勾配計算の結果の出力
x1.grad, x2.grad
```
	(tensor([16.]), tensor([36.]))



### 説明
