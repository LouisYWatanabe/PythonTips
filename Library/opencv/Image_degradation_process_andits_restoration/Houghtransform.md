# ハフ変換

名刺を抽出するのにハフ変換で直線を抽出します。

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
im = rgb2gray(imread('card.jpg'))

fig = plt.figure(figsize=(15,5))

fig.add_subplot(1, 2, 1)
imshow(im)
plt.axis('off')
plt.title('original image')

fig.add_subplot(1, 2, 2)
im_edge = canny(im, sigma=3, low_threshold=40/255, high_threshold=100/255)
imshow(im_edge)
plt.axis('off')
plt.title('Canny edge')

plt.show()
```


![png](output_45_0.png)



```python
angles = np.linspace(-np.pi / 2, np.pi / 2, 360)
voting_space, theta, rho = hough_line(im_edge, theta=angles)
```


```python
# 投票空間の可視化
imshow(np.log(1 + voting_space),
       extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), rho[-1], rho[0]],
       cmap='gray_r', aspect=0.08)
plt.xlabel('$\\theta$')
plt.ylabel('$\\rho$')
plt.title('voting space')
plt.show()
```


![png](output_47_0.png)



```python
# ハフ変換を行いピーク時のΘとρを取得
_, line_theta, line_rho = hough_line_peaks(voting_space, theta, rho, 
                                           threshold=0.3*voting_space.max(), num_peaks=4)
```


```python
x0 = 0
x1 = im.shape[1]
for angle, r in zip(line_theta, line_rho):
    y0 = (r - x0 * np.cos(angle)) / np.sin(angle)
    y1 = (r - x1 * np.cos(angle)) / np.sin(angle)
    plt.plot((x0, x1), (y0, y1), '-r')
imshow(im)
plt.title('lines found by hough transform')
plt.show()
```


![png](output_49_0.png)


ここから交点を抽出して、その4つのコーナーの正しい位置（右上とか左下）を判定します。
位置をもとに射影変換を求めて正方形に変換します。
正方形の画像から文字を認識します。
