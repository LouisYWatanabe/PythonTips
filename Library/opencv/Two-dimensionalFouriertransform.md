# 画像の2次元フーリエ変換

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
im = rgb2gray(imread('honeycomb.jpg'))

@interact(angle=(0, 360, 5))
def g(angle=0):

    fig = plt.figure(figsize=(10,5))
    
    fig.add_subplot(1, 2, 1)
    im_rot = rotate(im, angle=angle, preserve_range=True)
    imshow(im_rot)
    plt.axis('off')
    plt.title('original image')

    fig.add_subplot(1, 2, 2)
    # np.fft.fft2で二次元配列に対して二次元フーリエ変換を適用します。
    im_freq = np.fft.fft2(im_rot)
    h, w = im_freq.shape
#     im_freq = np.roll(im_freq, h//2, 0)
#     im_freq = np.roll(im_freq, w//2, 1)
    im_freq = np.fft.fftshift(im_freq)
    imshow(np.log10(np.abs(im_freq)) * 20, vmin=0)
    plt.axis('off')
    plt.title('power spectrum (log scale)')

    plt.show()
```

![png](./image/Two-dimensionalFouriertransform.png)

周波数成分は角度を持っているので画像を回転すると対応してフーリエスペクトルも回転します。
