# ガボール特徴も利用したkmeansクラスタリング

テクスチャ特徴量を利用して、髪や肌を認識させ、明暗で領域分割しないようにします。

ガボールフィルターの特徴をテクスチャ特徴量としてクラスタリングします。

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
img = rgb2gray(im)
```


```python
fig = plt.figure(figsize=(20,9))

n_i = 3
n_j = 10

for j in tqdm(range(n_i)):
    for i in tqdm(range(n_j), leave=False):
        ax = fig.add_subplot(n_i, n_j, i+1 + j*n_j)
        gabor_filter = gabor_kernel(frequency=0.1 * (j+1), bandwidth=1/(2*j+1), theta=0.4 * i).real
        imshow(gabor_filter)
        plt.tight_layout()

plt.show()
```


    HBox(children=(FloatProgress(value=0.0, max=3.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))


    



![png](output_27_5.png)



```python
fig = plt.figure(figsize=(20,9))

gabor_features = []

n_i = 3
n_j = 10

for j in tqdm(range(n_i)):
    for i in tqdm(range(n_j), leave=False):
        ax = fig.add_subplot(n_i, n_j, i+1 + j*n_j)
        im_gabor = gabor(img, frequency=0.1 * (j+1), bandwidth=1/(2*j+1), theta=0.4 * i)
        gabor_features.append(im_gabor[0]) # tuble (real, imag)
        imshow(im_gabor[0], cmap="gray")
        plt.axis('off')
        plt.tight_layout()

plt.show()
```


    HBox(children=(FloatProgress(value=0.0, max=3.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))



    HBox(children=(FloatProgress(value=0.0, max=10.0), HTML(value='')))


![png](output_28_5.png)


```python
gabor_texture = np.array(gabor_features).transpose(1,2,0).reshape((-1, n_i*n_j))
```


```python
X = np.hstack((im.reshape((-1, 3)), gabor_texture))
```


```python
X.mean(axis=0), X.std(axis=0)
```




    (array([ 5.16377106e-01,  5.51321589e-01,  5.76120282e-01,  5.25135480e-04,
             1.37765487e-03,  5.88044440e-04,  1.36045256e-03,  5.30336915e-04,
             1.38208281e-03,  6.02465921e-04,  1.33051405e-03,  5.45848007e-04,
             1.37510321e-03,  1.16334724e-04, -1.55378611e-06,  4.50891464e-06,
            -2.70187114e-07,  1.07389492e-04, -1.32017983e-06,  9.25559606e-06,
            -5.61885546e-07,  8.35464019e-05,  8.81302565e-08, -3.72636512e-05,
            -9.51397734e-08,  1.01658630e-05, -3.98535358e-07, -2.90043475e-05,
            -3.27676630e-07, -7.64132898e-07, -7.06667879e-08, -1.17115327e-05,
            -8.29450675e-07]),
     array([1.91432233e-01, 2.38683127e-01, 2.58553401e-01, 4.41122996e-03,
            4.05483165e-03, 4.07121432e-03, 5.24801666e-03, 5.58616571e-03,
            4.81801897e-03, 4.37466378e-03, 4.59841690e-03, 4.34447243e-03,
            4.01270439e-03, 9.84649868e-04, 7.99126754e-04, 7.25503191e-04,
            9.47366049e-04, 8.62365044e-04, 7.51550750e-04, 1.11434139e-03,
            1.12154042e-03, 8.46429991e-04, 7.85695158e-04, 4.42003659e-04,
            2.85745800e-04, 3.05797076e-04, 3.37127837e-04, 4.14152956e-04,
            2.52328065e-04, 2.93300564e-04, 5.41110582e-04, 3.76427240e-04,
            3.21853695e-04]))




```python
# 標準化
X -= X.mean(axis=0)
X /= X.std(axis=0)
```


```python
X.mean(axis=0), X.std(axis=0)
```




    (array([-7.59097521e-14,  1.46217657e-13,  2.42644734e-13, -4.88828397e-16,
             6.82444551e-16,  1.51065726e-15,  1.28791845e-15, -6.97798168e-16,
             8.79587736e-16,  1.08561170e-15,  4.28247398e-16,  4.82558249e-16,
            -8.57307614e-16,  2.91301143e-15, -2.99272205e-18,  4.39552753e-17,
             8.13988278e-18, -1.46599419e-15, -3.21185950e-17,  2.21641522e-17,
             4.63992360e-18, -1.66117897e-16,  3.27834404e-19,  1.21067090e-15,
             1.44135298e-17,  3.27232935e-16,  5.62319740e-18, -7.65762380e-16,
            -2.87837001e-17,  2.85876648e-17,  3.71010356e-18, -1.32436468e-16,
             3.63213391e-17]),
     array([1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.,
            1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]))




```python
clustering = KMeans(n_clusters=10)

# clustering.fit(X[::1000, :]) # 画素数を1/1000に間引き
clustering.fit(X)


result = clustering.predict(X)
img_seg = result.reshape(img.shape[:2])
```


```python
fig = plt.figure(figsize=(15,5))

fig.add_subplot(1, 2, 1)   
imshow(segmentation.mark_boundaries(im, img_seg, color=(1, 1, 1))) # 領域分割結果を境界線で表示
plt.axis('off')
plt.title('segmentation result')

fig.add_subplot(1, 2, 2)
imshow(color.label2rgb(img_seg)) # ラベリング結果をカラーで表示．
plt.axis('off')
plt.title('segmentation label')

plt.show()
```


![png](output_35_0.png)

