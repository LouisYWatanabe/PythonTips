# ソーベルフィルタ(Sobel)，プレウィットフィルタ(Prewitt), ロバーツフィルタ(Roberts)

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

@interact(val_max=(0.1, 0.5, 0.1))
def g(val_max=0.1):

    fig = plt.figure(figsize=(15, 3))

    ax = fig.add_subplot(1, 3, 1)
    imshow(sobel(im), vmin=0, vmax=val_max)
    plt.axis('off')
    plt.colorbar()
    plt.title('Sobel')

    ax = fig.add_subplot(1, 3, 2)
    imshow(prewitt(im), vmin=0, vmax=val_max)
    plt.axis('off')
    plt.colorbar()
    plt.title('Prewitt')

    ax = fig.add_subplot(1, 3, 3)
    imshow(roberts(im), vmin=0, vmax=val_max)
    plt.axis('off')
    plt.colorbar()
    plt.title('Roberts')

    plt.show()

print()
g(0.5)
```
![png](./image/sobelprewittroberts1.png)
![png](./image/sobelprewittroberts2.png)
