# 非線形フィルタ:バイラテラルフィルタ

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

@interact(sigma_spatial=(0, 15, 1), sigma_color=(0, 0.5, 0.1))
def g(sigma_spatial=1, sigma_color=0.1):
    fig = plt.figure(figsize=(15, 3))

    ax = fig.add_subplot(1, 3, 1)

    imshow(im)
    plt.axis('off')
    plt.title('original image')

    ax = fig.add_subplot(1, 3, 2)
    imshow(gaussian(im, sigma=sigma_spatial))
    plt.axis('off')
    plt.title('Gaussian filter with sigma={}'.format(sigma_spatial))
  
    ax = fig.add_subplot(1, 3, 3)
    im_denoise = denoise_bilateral(im,
                                   sigma_spatial=sigma_spatial, 
                                   sigma_color=sigma_color)
    imshow(im_denoise)
    plt.axis('off')
    plt.title('sigma_spatial={0} simga_color={1}'.format(sigma_spatial, sigma_color))

    plt.show()
```

![png](./image/Bilateralfilter.png)
