# Cannyエッジ

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

@interact(sigma=(0.1, 10, 0.1),
          th_low=(0, 255, 1),
          th_high=(0, 255, 1)
         )

def g(sigma=5, th_low=0, th_high=40):
    """
    Param:
    sigma: ガンマ関数の分散
    th_low: 閾値の下限値
    th_high: 閾値の上限値
    閾値の間に入った値を抽出します。
    """
    fig = plt.figure(figsize=(20,6))

    fig.add_subplot(1, 2, 1)
    imshow(im)
    plt.axis('off')
    plt.title('original image')
    
    fig.add_subplot(1, 2, 2)
    # canny検出の実行
    im_edge = canny(im, sigma=sigma, 
                    low_threshold=th_low/255, 
                    high_threshold=th_high/255)
    imshow(im_edge, cmap='gray_r')
    plt.axis('off')
    plt.title('Canny edge with th_low={0} and th_high={1}'.format(th_low, th_high))

    plt.show()

```

![png](./image/Canny1.png)

分散を大きくすると、大まかなエッジしか出てきません。
