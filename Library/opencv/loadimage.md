# 画像の読み込み
### 例

```python
'''
./data/test_image.jpg
を読み込む
'''
import cv2

image = cv2.imread('./data/test_image.jpg')

cv2.imshow('result', image)
cv2.waitKey(0)
```

![jpeg](./image/test_image.jpg)


```python
# jupyter notebook内で表示
import matplotlib.pyplot as plt
%matplotlib inline

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
```
