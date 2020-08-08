# テンプレートマッチング

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
im = imread('flag.png')[:, :, :3] # remove alpha channel

fig = plt.figure(figsize=(15,5))

fig.add_subplot(1, 2, 1)
imshow(im)
plt.title("original image")

fig.add_subplot(1, 2, 2)
# テンプレートとして画像を抽出
template = im[650:720, 843:960]
imshow(template, interpolation='none')
plt.title('template')

plt.show()
```


![png](output_42_0.png)



```python
@interact(angle=(-180, 180, 1))
def g(angle=0):
    
    im_rot = rotate(im, angle=angle, resize=False)

    ncc = match_template(rgb2gray(im_rot), rgb2gray(template))
    x, y = np.unravel_index(np.argmax(ncc), ncc.shape)[::-1] # y,x --> x,y

    fig = plt.figure(figsize=(15,5))

    ax = fig.add_subplot(1, 2, 1)

    imshow(im // 2)

    th, tw, _ = template.shape
    ax.add_patch(plt.Rectangle((x, y), tw, th, edgecolor='r', facecolor='none', lw=3))
    plt.title('detected template as rectangle')


    fig.add_subplot(1, 2, 2)
    imshow(ncc)
    plt.title('NCC')


    plt.show()
```


![png](output_43_0.png)

テンプレートマッチングでは角度が少しでもずれるとデータの取得に失敗します。
