# プログレスバーとファイルパスデータフレームの作成

# プログレスバーを表示する tqdm()

for文の処理経過を視覚的に表示します。


```python
from tqdm import tqdm
```


```python
# 1秒以上かかるループ処理
import time
before = time.time()
sum_ = 0

for i  in range(int(1e7)):
    sum_ += i
print(sum_)
after = time.time()
print('it took: {}'.format(after - before))
```

    49999995000000
    it took: 3.040893316268921


### tqdmの使用


```python
# 1秒以上かかるループ処理
import time
before = time.time()
sum_ = 0

# tqdmの使用
for i  in tqdm(range(int(1e7)), position=0):
    sum_ += i
print(sum_)
after = time.time()
print('it took: {}'.format(after - before))
```

    100%|██████████| 10000000/10000000 [00:08<00:00, 1206940.96it/s]

    49999995000000
    it took: 8.304287910461426


    


### DataFrameにtqdmを適用

'../../public-covid-data'のパスリストをそれぞれ作りデータフレームを作成します。

- path_im : '../../public-covid-data/rp_im/*.gz'
- filename : '*.gz'
- path_msk : '../../public-covid-data/rp_msk/*.gz'


```python
from glob import glob
import os
from pathlib import Path

import pandas as pd
```


```python
p = Path('../../public-covid-data')
path_folder = []
path_im = []
path_msk = []

filename_im = []
filename_msk = []

for folder in p.iterdir():
    path_folder.append(folder)
    file_head, file_tail = os.path.split(folder)
    
    if file_tail in 'rp_im':
        for im in folder.iterdir():
            # PosixPathの中のstringを取得
            path_im.append(str(im))
            # filenameの作成
            file_head, file_tail = os.path.split(im)
            filename_im.append(str(file_tail))
    elif file_tail in 'rp_msk':
        for im in folder.iterdir():
            # PosixPathの中のstringを取得
            path_msk.append(str(im))

            # filenameの作成
            file_head, file_tail = os.path.split(im)
            filename_msk.append(str(file_tail))
    
print(path_folder[0])
print()
print(path_im[0])
print()
print(path_msk[0])
print()
print(filename_im[0])
print()
print(len(filename_im))
```

    ../../public-covid-data/rp_im
    
    ../../public-covid-data/rp_im/1.nii.gz
    
    ../../public-covid-data/rp_msk/1.nii.gz
    
    1.nii.gz
    
    9



```python
# データフレームを作成
df_im = pd.DataFrame({'path':path_im, 'filename':filename_im})
df_msk = pd.DataFrame({'path':path_msk, 'filename':filename_msk})
```


```python
df_im.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
      <th>filename</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../../public-covid-data/rp_im/1.nii.gz</td>
      <td>1.nii.gz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../../public-covid-data/rp_im/2.nii.gz</td>
      <td>2.nii.gz</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../../public-covid-data/rp_im/3.nii.gz</td>
      <td>3.nii.gz</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../../public-covid-data/rp_im/4.nii.gz</td>
      <td>4.nii.gz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../../public-covid-data/rp_im/5.nii.gz</td>
      <td>5.nii.gz</td>
    </tr>
  </tbody>
</table>
</div>




```python
df_msk.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path</th>
      <th>filename</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../../public-covid-data/rp_msk/1.nii.gz</td>
      <td>1.nii.gz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../../public-covid-data/rp_msk/2.nii.gz</td>
      <td>2.nii.gz</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../../public-covid-data/rp_msk/3.nii.gz</td>
      <td>3.nii.gz</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../../public-covid-data/rp_msk/4.nii.gz</td>
      <td>4.nii.gz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../../public-covid-data/rp_msk/5.nii.gz</td>
      <td>5.nii.gz</td>
    </tr>
  </tbody>
</table>
</div>



データフレームのマージ


```python
# 'filename'をキーにしてマージ
df = df_im.merge(df_msk, on='filename', suffixes=('_im', '_msk'))
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>path_im</th>
      <th>filename</th>
      <th>path_msk</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>../../public-covid-data/rp_im/1.nii.gz</td>
      <td>1.nii.gz</td>
      <td>../../public-covid-data/rp_msk/1.nii.gz</td>
    </tr>
    <tr>
      <th>1</th>
      <td>../../public-covid-data/rp_im/2.nii.gz</td>
      <td>2.nii.gz</td>
      <td>../../public-covid-data/rp_msk/2.nii.gz</td>
    </tr>
    <tr>
      <th>2</th>
      <td>../../public-covid-data/rp_im/3.nii.gz</td>
      <td>3.nii.gz</td>
      <td>../../public-covid-data/rp_msk/3.nii.gz</td>
    </tr>
    <tr>
      <th>3</th>
      <td>../../public-covid-data/rp_im/4.nii.gz</td>
      <td>4.nii.gz</td>
      <td>../../public-covid-data/rp_msk/4.nii.gz</td>
    </tr>
    <tr>
      <th>4</th>
      <td>../../public-covid-data/rp_im/5.nii.gz</td>
      <td>5.nii.gz</td>
      <td>../../public-covid-data/rp_msk/5.nii.gz</td>
    </tr>
  </tbody>
</table>
</div>



ファイルパスのデータフレームを作ることは多く、このようにデータフレーム化することで、特定のデータに対して処理を行うことが容易になります。


```python
# filenameが'1.nii.gz'のデータパスを取得
print(df['path_im'][df['filename']=='1.nii.gz'])
print()
print(df['path_msk'][df['filename']=='1.nii.gz'])
```

    0    ../../public-covid-data/rp_im/1.nii.gz
    Name: path_im, dtype: object
    
    0    ../../public-covid-data/rp_msk/1.nii.gz
    Name: path_msk, dtype: object


## DataFrameにtqdmを適用し、for文処理の経過を確認する


```python
# DataFrameでイテレーションするときは.iterrows()を使用します
# Dataframeにtqdmを指定すると処理の総数から現在処理がどのくらい終了したが分かりません。
# 現在の処理経過を確認するためには`total=len(df)`を追加します。
for idx, rows in tqdm(df.iterrows(), total=len(df)):
    print('image path for {} is here {}'.format(rows['path_im'], rows['filename']))
```

    100%|██████████| 9/9 [00:00<00:00, 805.84it/s]

    image path for ../../public-covid-data/rp_im/1.nii.gz is here 1.nii.gz
    image path for ../../public-covid-data/rp_im/2.nii.gz is here 2.nii.gz
    image path for ../../public-covid-data/rp_im/3.nii.gz is here 3.nii.gz
    image path for ../../public-covid-data/rp_im/4.nii.gz is here 4.nii.gz
    image path for ../../public-covid-data/rp_im/5.nii.gz is here 5.nii.gz
    image path for ../../public-covid-data/rp_im/6.nii.gz is here 6.nii.gz
    image path for ../../public-covid-data/rp_im/7.nii.gz is here 7.nii.gz
    image path for ../../public-covid-data/rp_im/8.nii.gz is here 8.nii.gz
    image path for ../../public-covid-data/rp_im/9.nii.gz is here 9.nii.gz


    

