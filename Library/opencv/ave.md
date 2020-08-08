# 平均と分散の計算方法

### 定義式そのまま
- 計算量大：2重ループ2回
- 書いてはいけないコード

```python
im = rgb2gray(imread('girl.jpg'))
```

```python
h, w = im.shape

mean = 0
for y in range(h):
    for x in range(w):
        mean += im[y, x]
mean /= h * w
print('mean: ', mean)

var = 0
for y in range(h):
    for x in range(w):
        var += (im[y, x] - mean)**2
var /= h * w
print('variance: ', var)
print('std: ', np.sqrt(var))
```

    mean:  0.5493684053319258
    variance:  0.05135688987599337
    std:  0.2266205857286433

これは、悪いコードです。
二重ループが発生しているので処理に時間がかかり悪いコードです。

#### 定義式を変形したもの
- 計算量は半分：2重ループ1回
- 数式から導出できる
$
\mu = \frac{1}{N} \sum_i x_i \\
\sigma^2 = \frac{1}{N} \sum_i (x_i - \mu)^2
= (\frac{1}{N} \sum_i x_i^2) - \mu^2
$

```python
im = rgb2gray(imread('girl.jpg'))

h, w = im.shape

mean = 0
var = 0
for y in range(h):
    for x in range(w):
        mean += im[y, x]
        var  += im[y, x]**2

mean /= h * w
print('mean: ', mean)

var /= h * w
var -= mean**2
print('variance: ', var)
print('std: ', np.sqrt(var))
```
    mean:  0.5493684053319258
    variance:  0.051356889876493506
    std:  0.22662058572974678
