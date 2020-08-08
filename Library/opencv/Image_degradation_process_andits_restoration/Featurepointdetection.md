# 特徴点検出 （DoG・Fast・Harris・GFTT・AKAZE・BRISK・ORB）

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
im = imread('Colosseum.jpg')
img = rgb2gray(im)
imshow(im)
plt.show()
```


![png](output_51_0.png)


### DoG


```python
@interact(max_sigma=(10, 300, 10), 
          threshold=(0.02, 1, 0.02))
def g(max_sigma=50, threshold=0.2):
        
    keypoints1 = blob_dog(img, max_sigma=max_sigma, threshold=threshold, overlap=1)
    
    fig = plt.figure(figsize=(25,10))

    ax = fig.add_subplot(1, 2, 1)

    imshow(im)

    for k in keypoints1:
        y, x, s = k
        ax.add_patch(plt.Circle((x, y), s, edgecolor='r', facecolor='none', lw=1))

    plt.title('# keypoints {0}'.format(len(keypoints1)))
    plt.show()
```


![png](output_53_0.png)


### Fast


```python
@interact(n=(1, 16, 1), 
          threshold=(0.05, 0.5, 0.01))
def g(n=12, threshold=0.15):

    keypoints1 = corner_peaks(corner_fast(img, n=n, threshold=threshold))

    fig = plt.figure(figsize=(25,10))

    ax = fig.add_subplot(1, 2, 1)

    imshow(im)

    ax.scatter(keypoints1[:, 1], keypoints1[:, 0], color='r', marker='o', s=5)

    plt.title('# keypoints {0}'.format(len(keypoints1)))
    plt.show()
```


![png](output_55_0.png)


### Harris


```python
@interact(k=(0, 0.3, 0.01),
          sigma=(0.5, 3, 0.5))
def g(k=0.05, sigma=1):

    keypoints1 = corner_peaks(corner_harris(img, k=k, sigma=sigma))

    fig = plt.figure(figsize=(25,10))

    ax = fig.add_subplot(1, 2, 1)

    imshow(im)

    ax.scatter(keypoints1[:, 1], keypoints1[:, 0], color='r', marker='o', s=5)

    plt.title('# keypoints {0}'.format(len(keypoints1)))
    plt.show()
```


![png](output_57_0.png)


### GFTT


```python
@interact(maxCorners=(10,500,10),
          blockSize=(1,20,1))
def g(maxCorners=60, blockSize=3):

    keypoints1 = cv2.goodFeaturesToTrack(img.astype(np.float32),
                                         maxCorners=maxCorners,
                                         qualityLevel=0.01,
                                         minDistance=5,
                                         blockSize=blockSize)
    keypoints1 = np.squeeze(keypoints1)


    fig = plt.figure(figsize=(25,10))

    ax = fig.add_subplot(1, 2, 1)

    imshow(im)

    ax.scatter(keypoints1[:, 0], keypoints1[:, 1], color='r', marker='o', s=5)


    plt.title('# keypoints {0} {1}'.format(len(keypoints1), blockSize))
    plt.show()
```


![png](output_59_0.png)


### AKAZE


```python
@interact(descriptor_type=[3,5],
          threshold=(0.0001, 0.005, 0.0001))
def g(threshold=0.001, descriptor_type=3):

    detector = cv2.AKAZE_create(descriptor_type=descriptor_type, threshold=threshold)
    keypoints1, descriptors1 = detector.detectAndCompute(im, None)


    fig = plt.figure(figsize=(25,10))

    ax = fig.add_subplot(1, 2, 1)

    imshow(im)

    for k in keypoints1:
        x, y = k.pt
        s = k.size
        ax.add_patch(plt.Circle((x, y), s, edgecolor='r', facecolor='none', lw=1))


    plt.title('# keypoints {0}'.format(len(keypoints1)))
    plt.show()
```


![png](output_61_0.png)


### BRISK


```python
@interact(thresh=(10, 200, 10))
def g(thresh=100):

    detector = cv2.BRISK_create(thresh=thresh)
    keypoints1, descriptors1 = detector.detectAndCompute(im, None)


    fig = plt.figure(figsize=(25,10))

    ax = fig.add_subplot(1, 2, 1)

    imshow(im)

    for k in keypoints1:
        x, y = k.pt
        s = k.size
        ax.add_patch(plt.Circle((x, y), s, edgecolor='r', facecolor='none', lw=1))


    plt.title('# keypoints {0}'.format(len(keypoints1)))
    plt.show()
```


![png](output_63_0.png)


### ORB


```python
@interact(n_keypoints=(100, 2000, 100))
def g(n_keypoints=1000):

    descriptor_extractor = ORB(n_keypoints=n_keypoints)
    descriptor_extractor.detect_and_extract(img)
    keypoints1 = descriptor_extractor.keypoints # 特徴点の(y,x)座標
    # descriptors1 = descriptor_extractor.descriptors # 特徴量ベクトル

    fig = plt.figure(figsize=(25,10))

    ax = fig.add_subplot(1, 2, 1)

    imshow(im)

    ax.scatter(keypoints1[:, 1], keypoints1[:, 0], color='r', marker='o', s=5)

    plt.title('# keypoints {0}'.format(len(keypoints1)))
    plt.show()
```


![png](output_65_0.png)

