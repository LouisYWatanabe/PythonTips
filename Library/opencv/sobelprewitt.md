# ソーベルフィルタ，プレウィットフィルタ

```python
from scipy import ndimage
from scipy import signal
from scipy.misc import derivative


import skimage
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import gaussian, gabor_kernel, sobel, sobel_h, sobel_v, prewitt, prewitt_h, prewitt_v, roberts, median
from skimage.io import imread, imsave
from skimage.restoration import denoise_bilateral, denoise_nl_means
from skimage.transform import rotate, resize
from skimage.morphology import square


import matplotlib.pyplot as plt
%matplotlib inline
plt.gray();
from matplotlib.pyplot import imshow
import matplotlib.mlab as mlab
import matplotlib.colors as colors

import numpy as np
from numpy.fft import fft

import wave

from time import time


import ipywidgets as widgets
from IPython.display import display
from ipywidgets import interact, interactive, fixed, RadioButtons

from tqdm.notebook import tqdm
```

```python
im = rgb2gray(imread('girl.jpg'))
# 計算の短縮のためフィルタサイズを縮小
im = resize(im, (im.shape[0]//5, im.shape[1]//5))
# 関数はあるが、フィルターカーネルを用意
# 'diff'は横方向の中心差分
# 係数で割っています。
kernels = {'diff': np.array([[ 0,0,0],
                             [-1,0,1],
                             [ 0,0,0]]) / 2,
           'prewitt': np.array([[-1,0,1],
                                [-1,0,1],
                                [-1,0,1]]) / 6,
           'sobel': np.array([[-1,0,1],
                              [-2,0,2],
                              [-1,0,1]]) / 8
          }

print('diff filter')

@interact(kernel=['diff', 'prewitt', 'sobel'],
          val_max=(0.1, 0.5, 0.1))

def g(kernel='diff', val_max=0.1):

    k = kernels[kernel]
    # 横方向の微分を計算
    imh = ndimage.convolve(im, k)
    # 縦方向の微分を計算
    imv = ndimage.convolve(im, k.T)

    fig = plt.figure(figsize=(15, 3))

    ax = fig.add_subplot(1, 3, 1)
    imshow(imh, cmap="bwr", vmin=-0.5, vmax=0.5)
    plt.axis('off')
    plt.colorbar()
    plt.title('$I_x$')

    ax = fig.add_subplot(1, 3, 2)
    imshow(imv, cmap="bwr", vmin=-0.5, vmax=0.5)
    plt.axis('off')
    plt.colorbar()
    plt.title('$I_y$')

    ax = fig.add_subplot(1, 3, 3)
    imshow(np.sqrt(imv**2 + imh**2), cmap="gray", vmin=0, vmax=val_max)
    plt.axis('off')
    plt.colorbar()
    plt.title('$\sqrt{I_x^2 + I_y^2}$')

    plt.show()
    
print()

print('Sobel filter')
g(kernel='sobel', val_max=0.1)
print()

print('prewitt filter')
g(kernel='prewitt', val_max=0.1)
print()


g(kernel='diff', val_max=0.5)
print()

print('Sobel filter')
g(kernel='sobel', val_max=0.5)
print()

print('prewitt filter')
g(kernel='prewitt', val_max=0.5)
```

    diff filter
![png](./image/sobelprewitt1.png)
    Sobel filter
![png](./image/sobelprewitt2.png)
    prewitt filter
![png](./image/sobelprewitt3.png)
    diff filter
![png](./image/sobelprewitt4.png)
    Sobel filter
![png](./image/sobelprewitt5.png)
    prewitt filter
![png](./image/sobelprewitt6.png)

表示される画像は
- 横方向のフィルターの縦方向エッジ
- 縦方向のフィルターの横方向エッジ
- 上記の絶対値を取ったエッジ

を示します。