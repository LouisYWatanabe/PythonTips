# ファイルのパスリストの取得・操作 glob() os pathlib.Path

[http://medicalsegmentation.com/covid19/](http://medicalsegmentation.com/covid19/)のSegmentation dataset nr. 2 (13th April):から取得できる「Image volumes (308 Mb) – 9 volumes, total of >800 slices」「Covid19 masks (0.3 Mb) – includes >350 annotated slices」データを使用してファイルのリスト操作を見てみます


```python
from glob import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
#引数のパターンにマッチするファイルパスのリストを取得
# * : ワイルドカード(0文字以上の任意の文字列)
glob('../../public-covid-data/*/*')
```




    ['../../public-covid-data/rp_im/1.nii.gz',
     '../../public-covid-data/rp_im/2.nii.gz',
     '../../public-covid-data/rp_im/3.nii.gz',
     '../../public-covid-data/rp_im/4.nii.gz',
     '../../public-covid-data/rp_im/5.nii.gz',
     '../../public-covid-data/rp_im/6.nii.gz',
     '../../public-covid-data/rp_im/7.nii.gz',
     '../../public-covid-data/rp_im/8.nii.gz',
     '../../public-covid-data/rp_im/9.nii.gz',
     '../../public-covid-data/rp_msk/1.nii.gz',
     '../../public-covid-data/rp_msk/2.nii.gz',
     '../../public-covid-data/rp_msk/3.nii.gz',
     '../../public-covid-data/rp_msk/4.nii.gz',
     '../../public-covid-data/rp_msk/5.nii.gz',
     '../../public-covid-data/rp_msk/6.nii.gz',
     '../../public-covid-data/rp_msk/7.nii.gz',
     '../../public-covid-data/rp_msk/8.nii.gz',
     '../../public-covid-data/rp_msk/9.nii.gz']




```python
# 5~9にマッチしたファイルを取得
glob('../../public-covid-data/rp_im/[5-9]*')
```




    ['../../public-covid-data/rp_im/5.nii.gz',
     '../../public-covid-data/rp_im/6.nii.gz',
     '../../public-covid-data/rp_im/7.nii.gz',
     '../../public-covid-data/rp_im/8.nii.gz',
     '../../public-covid-data/rp_im/9.nii.gz']



## osとpathlib

### pathlib.Path

イテレーターとして指定されたパスのパスオブジェクトを返します。<br>イテレーターをイテレーターとして作成することでさらに深いパスを取得できます。


```python
p = Path('../../public-covid-data')
```


```python
p
```




    PosixPath('../../public-covid-data')




```python
# イテレーターの作成
p_i = p.iterdir()
```


```python
p_i_ = next(p_i)
p_i_
```




    PosixPath('../../public-covid-data/rp_im')




```python
next(p_i)
```




    PosixPath('../../public-covid-data/rp_msk')




```python
p_i_i = p_i_.iterdir()
```


```python
next(p_i_i)
```




    PosixPath('../../public-covid-data/rp_im/1.nii.gz')




```python
p = Path('../../public-covid-data')
```


```python
# イテレーターの中のリストを取得
sub_p = list(p.iterdir())
sub_p
```




    [PosixPath('../../public-covid-data/rp_im'),
     PosixPath('../../public-covid-data/rp_msk')]




```python
sub_p = sub_p[0]
sub_p
```




    PosixPath('../../public-covid-data/rp_im')




```python
list(sub_p.iterdir())
```




    [PosixPath('../../public-covid-data/rp_im/1.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/2.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/3.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/4.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/5.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/6.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/7.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/8.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/9.nii.gz')]



Pathオブジェクトにもglob()を使用することが可能です。


```python
list(sub_p.glob('*[6-9]*'))
```




    [PosixPath('../../public-covid-data/rp_im/6.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/7.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/8.nii.gz'),
     PosixPath('../../public-covid-data/rp_im/9.nii.gz')]



#### pathオブジェクトに対して実行する処理

### os.path()


```python
target_file = list(sub_p.glob('*'))[0]
target_file
```




    PosixPath('../../public-covid-data/rp_im/1.nii.gz')




```python
file_head, file_tail = os.path.split(target_file)
print(file_head)
print(file_tail)
```

    ../../public-covid-data/rp_im
    1.nii.gz



```python
os.path.join(file_head, file_tail)
```




    '../../public-covid-data/rp_im/1.nii.gz'



#### フォルダの作成

`../../public-covid-data/`に新しく`new_folder`を作成します。


```python
# フォルダ作成する場所のパス
p = Path('../../public-covid-data')
# 作成するフォルダ名
new_folder_name = 'new_folder'
# 作成するフォルダのパス
new_folder_path = os.path.join(p, new_folder_name)
new_folder_path
```




    '../../public-covid-data/new_folder'



#### os.makedirs()

フォルダを作成します。<br>すでにフォルダがあるとエラーになるため`if not os.path.exists()`を使用することが多いです

#### os.path.exists()

同じパスがあるかどうかを判定します。


```python
os.path.exists(new_folder_path)
```




    False




```python
# フォルダの作成
if not os.path.exists(new_folder_path):
    # パスがなければフォルダを作成
    os.makedirs(new_folder_path)
```
