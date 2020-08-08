# 動的輪郭モデル

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
img = rgb2gray(imread('coins.jpg'))
imshow(img)
img_s = gaussian(img, 3)
```


![png](output_37_0.png)



```python
# #### image boundary
# h, w = img.shape[:3]
# margin = 5
# N = 50
# y = \
# np.concatenate((np.array([margin]*N),
#                 np.linspace(margin, h-1-margin, N),
#                 np.array([h-1-margin]*N),
#                 np.linspace(margin, h-1-margin, N)))
# x = \
# np.concatenate((np.linspace(margin, w-1-margin, N),
#                 np.array([w-1-margin]*N),
#                 np.linspace(w-1-margin, margin, N),
#                 np.array([margin]*N),
#                ))

#### circle
s = np.linspace(0, 2*np.pi, 100)
y = 100 + 80 * np.sin(s)
x = 100 + 80 * np.cos(s)
```


```python
snake = np.stack((y, x), axis=1)

all_snake = [snake]
for i in range(100):
    snake = active_contour(img_s,
                           snake,
#                            alpha=0.01, beta=5,
                           alpha=0.02, beta=5,
#                            alpha=0.05, beta=5,
#                            alpha=0.1, beta=5,
                           max_iterations=10,
                           coordinates='rc')
    all_snake.append(snake)

@interact(itr=(0, len(all_snake)-1, 1))
def g(itr=0):
    imshow(img)
    plt.plot(all_snake[0][:, 1], all_snake[0][:, 0], '.w', lw=5)
    plt.plot(all_snake[itr][:, 1], all_snake[itr][:, 0], '.r', lw=5)
    plt.axis('off')
    plt.show()
```


![png](output_39_0.png)



```python
for i in range(1, 20, 2):
    g(itr=i)
```


![png](output_40_0.png)



![png](output_40_1.png)



![png](output_40_2.png)



![png](output_40_3.png)



![png](output_40_4.png)



![png](output_40_5.png)



![png](output_40_6.png)



![png](output_40_7.png)



![png](output_40_8.png)



![png](output_40_9.png)

