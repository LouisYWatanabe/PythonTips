# Autograd: 自動微分

### 例



```python
from __future__ import print_function
import torch

print(torch.__version__)
```

    1.5.0+cpu
    

テンソルを作成する際、引数に`requires_grad = True`を設定することでテンソルで勾配が記録されるようになります。

`backward() `で勾配を計算すると、Tensor の grad 属性に勾配が保持されます。


```python
x = torch.ones(2, 2, requires_grad=True)
print(x)
```

    tensor([[1., 1.],
            [1., 1.]], requires_grad=True)
    


```python
# 計算グラフ（式） y を作成します。
y = x + 2
print(y)
print(y.grad_fn)
```

    tensor([[3., 3.],
            [3., 3.]], grad_fn=<AddBackward0>)
    <AddBackward0 object at 0x00000188A4DFD7F0>
    

``y``は操作の結果として、``grad_fn``が作成されます。

`y` を利用してさらに計算グラフ（式） `z`、 `out` を作成します。


```python
z = y * y * 3
out = z.mean()
print(z, out)
```

    tensor([[27., 27.],
            [27., 27.]], grad_fn=<MulBackward0>) tensor(27., grad_fn=<MeanBackward0>)
    

### 勾配

`out.backward() `で勾配を計算します。


```python
out.backward()
```

out の x による偏微分、$\frac{d(out)}{dx} $を出力します。


```python
print(x.grad)
```

    tensor([[4.5000, 4.5000],
            [4.5000, 4.5000]])
    

out は out = z.mean() 、 z は z = y * y * 3 ですので、以下の式になります。

$out = \frac{1}{4}\sum_i z_i$,

$z_i = 3(x_i+2)^2$ and 

$z_i\bigr\rvert_{x_i=1} = 27$

よって、outをxで偏微分すると

$\frac{\partial out}{\partial x_i} = \frac{3}{2}(x_i+2)$, 

$\frac{\partial out}{\partial x_i}\bigr\rvert_{x_i=1} = \frac{9}{2} = 4.5$.



### 説明
