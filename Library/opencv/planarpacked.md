# 3次元配列でのカラー画像の表現

3次元配列のカラー画像はチャネルファーストとチャネルラストという2つの表現方法があります。

### Packed format

- 画素へのアクセスの順番
    - 縦、横、チャネル
    - y, x, color
    - 1画素の色情報のメモリ領域が連続している
- チャネルラスト：一般的な場合はこれ

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
im = np.zeros((2, 4, 3), dtype=np.uint8)    # 縦2，横4，3チャンネルを0で作成

im[0, 1, 0] = 255

print(im[:, :, 0])    # 赤：0番目のチャネル
print(im[:, :, 1])    # 緑：1番目のチャネル
print(im[:, :, 2])    # 青：2番目のチャネル

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

### Planar format

- 画素へのアクセスの順番
- チャネル、縦、横
    - channel, y, x
    - 「2x3の画像」のメモリ領域が連続している
- 特殊用途
    - 一部の動画フォーマット
    - 深層学習では「バッチ」

```python
im = np.zeros((3, 2, 4), dtype=np.uint8) # 画像を3枚，縦2，横4
im[0, 0, 1] = 255
print(im)
```
    [[[  0 255   0   0]
    [  0   0   0   0]]

    [[  0   0   0   0]
    [  0   0   0   0]]

    [[  0   0   0   0]
    [  0   0   0   0]]]

```python
imshow(im) # imshowはpacked formatを仮定しているので，このplanar formatを表示するとおかしなことになる
plt.axis('off')
plt.show()
```

![png](./image/planar2.png)
