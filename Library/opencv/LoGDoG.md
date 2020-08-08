# Laplacian-of-GaussianとDifference-of-Gaussian

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

## LoG

```python
def gauss(x, sigma=1):
    return np.exp(- x**2 / 2 / sigma**2) / 2 / np.pi / sigma

def grad_gauss(x, sigma, n=1):
    return derivative(gauss, x, dx=1e-6, n=n, args=({sigma:sigma})) # n次導関数を計算

@interact(sigma=(0.1, 2, 0.05))
def g(sigma=1):
    
    x = np.arange(-5, 5, 0.1)
    plt.plot(x, gauss(x, sigma=sigma), label="f(x)")
    plt.plot(x, grad_gauss(x, sigma=sigma), label="f'(x)")
    plt.plot(x, grad_gauss(x, sigma=sigma, n=2), label="f''(x)")
    plt.title("Gauss f(x) and derivatives f'(x), f''(x)")
    plt.xlabel("x")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()

g()
```

![png](./image/LGD.png)
![png](./image/LGD1.png)

一次導関数がオレンジ、二次導関数が緑で表されます。
ラプラシアンガウシアンのフィルターはメキシカンハット型になっていることが分かります。

## DoG

```python
@interact(sigma1=(0.1, 2, 0.05),
         sigma2=(0.1, 2, 0.05))
def g(sigma1=1,sigma2=2):
    
    x = np.arange(-5, 5, 0.1)
    plt.plot(x, gauss(x, sigma=sigma1), label="f1(x)")
    plt.plot(x, gauss(x, sigma=sigma2), label="f2(x)")
    plt.plot(x, gauss(x, sigma=sigma1) - gauss(x, sigma=sigma2), label="f1 - f2")
    plt.title("f1(x), f2(x), and f1(x) - f2(x)")
    plt.xlabel("x")
    plt.ylim(-0.5, 0.5)
    plt.legend()
    plt.show()

g()
```

![png](./image/LGD2.png)
![png](./image/LGD3.png)
