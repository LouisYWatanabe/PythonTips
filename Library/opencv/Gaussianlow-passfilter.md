# ガウス型ローパスフィルタ

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

# インパルス応答を作成
impulse = np.ones(im.shape) * np.finfo(np.float32).eps # avoid 0-division in log
h, w = im.shape
impulse[h//2, w//2] = 1

@interact(sigma=(1, 50, 5))
def g(sigma=3):

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
    imshow(np.log10(np.abs(im_freq) * 20), vmin=0)
    plt.axis('off')
    plt.title('fourier spectrum')

    im_freq2 = im_freq.copy()
    im_freq2 *= gaussian(impulse, sigma=sigma)
    fig.add_subplot(1, 4, 3)
    imshow(np.log10(np.abs(im_freq2) * 20), vmin=0)
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

![png](./image/Gaussianlow-passfilter.png)

リンギングなくぼかしができています。
