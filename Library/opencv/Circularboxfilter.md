# 円形ボックスフィルタ

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

@interact(radius=(0, 200, 5))
def g(radius=30):

    fig = plt.figure(figsize=(20,5))

    fig.add_subplot(1, 4, 1)
    imshow(im)
    plt.axis('off')
    plt.title('original image')

    fig.add_subplot(1, 4, 2)
    im_freq = np.fft.fft2(im)
    h, w = im_freq.shape
#     im_freq = np.roll(im_freq, h//2, 0)
#     im_freq = np.roll(im_freq, w//2, 1)
    im_freq = np.fft.fftshift(im_freq)
    imshow(np.log10(np.abs(im_freq)) * 20, vmin=0)
    plt.axis('off')
    plt.title('fourier spectrum')

    im_freq2 = im_freq.copy()
    im_freq2 *= 0.0001
    # 中央の座標を中心にした半径radiusの円の座標を作成
    rr, cc = skimage.draw.circle(h//2, w//2, radius)
    # 周波数成分が円座標に対応する所だけ抽出
    im_freq2[rr, cc] = im_freq[rr, cc]
    fig.add_subplot(1, 4, 3)
    imshow(np.log10(np.abs(im_freq2)) * 20, vmin=0)
    plt.axis('off')
    plt.title('filtered spectrum')
    
    fig.add_subplot(1, 4, 4)
#     im_freq2 = np.roll(im_freq2, h//2, 0)
#     im_freq2 = np.roll(im_freq2, w//2, 1)
    im_freq2 = np.fft.fftshift(im_freq2)
    g = np.fft.ifft2(im_freq2)
    imshow(np.abs(g))
    plt.axis('off')
    plt.title('filtered image')

    plt.show()
```

![png](./image/Circularboxfilter.png)

元画像が左の画像で横の画像がフーリエスペクトル、（逆フーリエ変換で使用する）マスク、逆フーリエ変換結果という表示内容です。
この結果からマスクがフーリエスペクトルよりかなり小さくても画像は復元できることが分かります。
つまりフーリエスペクトルのほとんどは必要ないことが分かります。
自然画像はたいてい、低周波成分だけでできています。
