# PyTorchテンソル ←→ NumPy多次元配列値、の変換＆連携

### 書式&チートシート

```python
import torch
import numpy as np
# PyTorchテンソルを、NumPy多次元配列に変換
b = x.numpy()    # 「numpy()」を呼び出すだけ。以下は注意点（メモリ位置の共有）

# ※PyTorchテンソル側の値を変えると、NumPy多次元配列値「b」も変化する（トラックされる）
print ('PyTorch計算→NumPy反映：')
print(b); x.add_(y); print(b)           # PyTorch側の計算はNumPy側に反映
print ('NumPy計算→PyTorch反映：')
print(x); np.add(b, b, out=b); print(x) # NumPy側の計算はPyTorch側に反映

# -----------------------------------------
# NumPy多次元配列を、PyTorchテンソルに変換
c = np.ones((2, 3), dtype=np.float64) # 2行3列の多次元配列値（1で初期化）を生成
d = torch.from_numpy(c)  # 「torch.from_numpy()」を呼び出すだけ

# ※NumPy多次元配列値を変えると、PyTorchテンソル「d」も変化する（トラックされる）
print ('NumPy計算→PyTorch反映：')
print(d); np.add(c, c, out=c); print(d)  # NumPy側の計算はPyTorch側に反映
print ('PyTorch計算→NumPy反映：')
print(c); d.add_(y); print(c)            # PyTorch側の計算はNumPy側に反映
```
	PyTorch計算→NumPy反映：
	[[2.  2.1 2.2]
	[3.  3.1 3.2]]
	[[3.  3.1 3.2]
	[4.  4.1 4.2]]
	NumPy計算→PyTorch反映：
	tensor([[3.0000, 3.1000, 3.2000],
		[4.0000, 4.1000, 4.2000]])
	tensor([[6.0000, 6.2000, 6.4000],
		[8.0000, 8.2000, 8.4000]])
	NumPy計算→PyTorch反映：
	tensor([[1., 1., 1.],
		[1., 1., 1.]], dtype=torch.float64)
	tensor([[2., 2., 2.],
		[2., 2., 2.]], dtype=torch.float64)
	PyTorch計算→NumPy反映：
	[[2. 2. 2.]
	[2. 2. 2.]]
	[[3. 3. 3.]
	[3. 3. 3.]]