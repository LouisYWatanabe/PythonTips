# ガウシアンフィルタ

```python
# ガウシアンフィルタ
blur = cv2.GaussianBlur(gray, (5, 5), 0)
```

### 例

```python
import cv2

image = cv2.imread('./data/test_image.jpg')

import matplotlib.pyplot as plt
%matplotlib inline

# グレースケール
gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
# plt.imshow(cv2.cvtColor(gray, cv2.COLOR_BGR2RGB))

# ガウシアンフィルタ（5x5カーネルのグレースケール画像）
blur = cv2.GaussianBlur(gray, (5, 5), 0)

plt.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
```

![jpeg](./image/GaussianBlur.png)


```python
cv2.imshow('result', blur)
cv2.waitKey(0)
```
