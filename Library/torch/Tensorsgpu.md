# テンソル演算でのGPU利用

### 書式&チートシート

```python
import torch
import numpy as np
# NVIDIAのGPUである「CUDA」（GPU）デバイス環境が利用可能な場合、
# GPUを使ってテンソルの計算を行うこともできる
if torch.cuda.is_available():              # CUDA（GPU）が利用可能な場合
    print('CUDA（GPU）が利用できる環境')
    print(f'CUDAデバイス数： {torch.cuda.device_count()}')
    print(f'現在のCUDAデバイス番号： {torch.cuda.current_device()}')  # ※0スタート
    print(f'1番目のCUDAデバイス名： {torch.cuda.get_device_name(0)}') # 例「Tesla T4」   

    device = torch.device("cuda")          # デフォルトのCUDAデバイスオブジェクトを取得
    device0 = torch.device("cuda:0")       # 1番目（※0スタート）のCUDAデバイスを取得

    # テンソル計算でのGPUの使い方は主に3つ：
    g = torch.ones(2, 3, device=device)    # （1）テンソル生成時のパラメーター指定
    g = x.to(device)                       # （2）既存テンソルのデバイス変更
    g = x.cuda(device)                     # （3）既存テンソルの「CUDA（GPU）」利用
    f = x.cpu()                            # （3'）既存テンソルの「CPU」利用

    # ※（2）の使い方で、GPUは「.to("cuda")」、CPUは「.to("cpu")」と書いてもよい
    g = x.to("cuda")
    f = x.to("cpu")

    # ※（3）の引数は省略することも可能
    g = x.cuda()

    # 「torch.nn.Module」オブジェクト（model）全体でのGPU／CPUの切り替え
    model.cuda()  # モデルの全パラメーターとバッファーを「CUDA（GPU）」に移行する
    model.cpu()   # モデルの全パラメーターとバッファーを「CPU」に移行する
else:
    print('CUDA（GPU）が利用できない環境')
```
	CUDA（GPU）が利用できる環境
	CUDAデバイス数： 1
	現在のCUDAデバイス番号： 0
	1番目のCUDAデバイス名： Tesla K80

### 説明

ColabでGPUを有効にするには、メニューバーの［ランタイム］－［ランタイムのタイプを変更］をクリックして切り替えること