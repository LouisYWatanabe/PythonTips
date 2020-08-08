# RGBとBGR

### RGB
- 多くの教科書の説明もこれ
- 多くの画像処理ライブラリはこれ
    - pythonならskimage, matplotlib

```python
import cv2

import skimage
from skimage.io import imread, imsave

from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.filters import threshold_otsu, threshold_local
from skimage.exposure import histogram, adjust_gamma
from skimage.morphology import square
from skimage import measure, color, morphology
# 警告の非表示
import warnings
warnings.filterwarnings('ignore')

im = np.zeros((2, 4, 3), dtype=np.uint8) # 縦2，横4，3チャネル分

im[0, 1, 0] = 255

print(im[:, :, 0]) # 赤：0番目のチャネル
print(im[:, :, 1]) # 緑：1番目のチャネル
print(im[:, :, 2]) # 青：2番目のチャネル

imshow(im)
plt.axis('off')
plt.show()
```
    [[  0 255   0   0]
    [  0   0   0   0]]
    [[0 0 0 0]
    [0 0 0 0]]
    [[0 0 0 0]
    [0 0 0 0]]

![png](./image/planar.png)

### BGR
- こちらもよく使われる
    - opencv（python, C/C++）
    - WindowsのCOLORREF（16進で0x00bbggrr）
    - ハードウェア
- データの解釈の違いだけ

```python
im = np.zeros((2, 4, 3), dtype=np.uint8)

im[0, 1, 0] = 255

print(im[:, :, 0]) # 青：0番目のチャネル
print(im[:, :, 1]) # 緑：1番目のチャネル
print(im[:, :, 2]) # 赤：2番目のチャネル
```
    [[  0 255   0   0]
    [  0   0   0   0]]
    [[0 0 0 0]
    [0 0 0 0]]
    [[0 0 0 0]
    [0 0 0 0]]

```python
imshow(im) # このmatplotlibのimshowはRGBを仮定
plt.axis('off')
plt.show()
```
![png](./image/planar.png)

```python
# opencvのimshowはBGRを仮定
cv2.imshow('opencv imshow window', cv2.resize(im, (400, 200), interpolation=cv2.INTER_NEAREST))
cv2.waitKey(3000)  # 3000ms（3秒）待つ
cv2.destroyWindow('opencv imshow window') # 消えないかもしれないけど無視
```
