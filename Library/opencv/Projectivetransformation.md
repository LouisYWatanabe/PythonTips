# 射影変換

```python
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
plt.gray();
from matplotlib.pyplot import imshow

import skimage
from skimage.io import imread, imsave
from skimage.transform import rotate, resize
from skimage.filters import gaussian
from skimage.transform import AffineTransform, ProjectiveTransform, warp

from ipywidgets import interact, interactive, fixed, RadioButtons
import ipywidgets as widgets
from IPython.display import display
```

```python
im = imread('girl.jpg')


H = np.array([[1.0, 0.01,  10], 
              [0.01, 1.0,  20], 
              [-0.001, 0.002,  1]])
# warpで元画像、射影変換オブヘクトを指定します
# 射影変換には逆変換を指定
imshow(warp(im, ProjectiveTransform(H).inverse))
plt.show()
```


![png](./image/output_6_0.png)
