# ウィーナフィルタによる画像の復元

```python
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
plt.gray();
from matplotlib.pyplot import imshow
import matplotlib.colors as colors


import skimage
from skimage import color, data, filters, restoration, morphology, measure, segmentation
from skimage.io import imread, imsave
from skimage.color import rgb2gray, rgb2hsv
from skimage.transform import rotate, resize
from skimage.transform import AffineTransform, ProjectiveTransform, warp 
from skimage.transform import hough_line, hough_line_peaks
from skimage.filters import gaussian, gabor_kernel, gabor
from skimage.feature import canny, match_template
from skimage.feature import corner_harris, corner_fast, blob_dog, ORB
from skimage.feature import match_descriptors, corner_peaks, plot_matches, corner_subpix
from sklearn.cluster import KMeans, MeanShift
from skimage.measure import ransac
from skimage.segmentation import active_contour

import scipy as sp
from scipy import ndimage
from scipy import signal
from scipy import fft

from time import time

import cv2

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

from ipywidgets import interact, interactive, fixed, RadioButtons
import ipywidgets as widgets
from IPython.display import display

from tqdm.notebook import tqdm
```
```python
im = rgb2gray(imread('girl.jpg'))


@interact(noise=(0, 1, 0.1), 
          balance=(1, 50, 1),
          sigma=(1, 20, 1))
def g(noise=0.3, sigma=10, balance=10):
    fig = plt.figure(figsize=(20, 3))

    fig.add_subplot(1, 4, 1)
    imshow(im)
    plt.axis('off')
    plt.title('original image')

    fig.add_subplot(1, 4, 2)
    impulse = np.zeros((sigma*5, sigma*5)) # use five sigma
    h, w = impulse.shape
    impulse[h//2, w//2] = 1
    psf = gaussian(impulse, sigma=sigma)
    imshow(psf)
    plt.title('PSF: sigma={}'.format(sigma))

    fig.add_subplot(1, 4, 3)
    img = signal.fftconvolve(im, psf, mode='same')
    img += noise * img.std() * np.random.standard_normal(img.shape) # 画像をPFSでぼかしてノイズを加える
    imshow(img)
    plt.axis('off')
    plt.title('observed image')

    fig.add_subplot(1, 4, 4)
    deconvolved_img = restoration.wiener(img, psf, balance)
    imshow(deconvolved_img, vmin=0, vmax=1)
    plt.axis('off')
    plt.title('restored image, balance {}'.format(balance))

    plt.show()
```

![png](./image/Wienerfilter.png)
