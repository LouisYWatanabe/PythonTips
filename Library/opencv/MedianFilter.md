# 非線形フィルタ:メディアンフィルタ

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
im = imread('salt_and_pepper.png')

@interact(sigma=(0, 10, 1), N=(1, 10, 1))
def g(sigma=2, N=3):

    fig = plt.figure(figsize=(15, 5))

    ax = fig.add_subplot(1, 3, 1)
    imshow(im)
    plt.axis('off')
    plt.title('original image')

    ax = fig.add_subplot(1, 3, 2)
    imshow(gaussian(im, sigma=sigma))
    plt.axis('off')
    plt.title('Gaussian filter with $\sigma$={}'.format(sigma))

    ax = fig.add_subplot(1, 3, 3)
    # メディアンフィルタの実行
    imshow(median(im, square(N)))
    plt.axis('off')
    plt.title('Median filter with {0}x{0} patch'.format(N))

    plt.show()
```

![png](./image/MedianFilter.png)

ガウスフィルターを適用してもボケるだけでノイズが消えないようなときにメディアンフィルターを使用します。
メディアンフィルターはこのプログラムでは周辺3×3のフィルターの中の中央値を出力する方法でぼかし処理を行っています。
