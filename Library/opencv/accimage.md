# 画像へのアクセス;行列

- 配列へのアクセスの順番
    - 行，列
- 画素へのアクセスの順番
    - 縦，横
    - y, x
- ループを回すなら外側がy，内側がx
    - 配列2つ目のインデックスのほうが連続したメモリ領域

### 例

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

# 幅3✕高さ2で0の画素の画像を作り、(1, 0)の画素を255にする
im = np.zeros((2, 3))    # 幅3✕高さ2の画像（配列）
im[0, 1] = 255           # (x,y)=(1,0)の画素へアクセス
print(im)

imshow(im)    # 画像の表示
plt.axis('off')    # メモリを非表示
plt.show()
```
    [[  0. 255.   0.]
    [  0.   0.   0.]]

![png](./image/accimage.png)