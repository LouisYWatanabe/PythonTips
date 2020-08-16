# 並列処理(multiprocessing)


```python
from multiprocessing import Pool, cpu_count
```


```python
import numpy as np
import time
from tqdm import tqdm
```

## map

`map(func, iterables)`は、iterablesが返すそれぞれの値に対してfunc関数を実行できます。

`map()`の練習用に関数を作成してmapを実行してみます。


```python
def square(n):
    return n **2

params = np.arange(1, 5)

# イテレータの取得
i = map(square, params)
# 表示の確認
[print(next(i)) for _ in params]
```

    1
    4
    9
    16





    [None, None, None, None]




```python
# map()を使用しない場合
[square(p) for p in params]
```




    [1, 4, 9, 16]



`map()`関数は、内包表現で簡単に補うことができるので通常使用しないですが、<br>並列処理では使用します。

## Pool.map() Pool.imap()

指定した秒数処理を行わない関数`wait_sec()`作成し、<br>並列処理の有無を処理時間を比較することで確認します。


```python
# Pool()に指定するため物理コアの数を指定します
cpu_count()    # 物理コアの数の取得
```




    3




```python
# 物理コアの数-1を使用する物理コアの数としてPoolに指定
p = Pool(processes=cpu_count()-1)
# multiprocessを閉じる
p.close()
p.join()

def wait_sec(sec):
    # sec秒間処理を待機
    time.sleep(sec)
    return sec ** 2
```

`p = Pool(processes=cpu_count()-1)`を使用した場合は、使用後にmultiprocessを閉じた方がよいです。<br>閉じない場合pのインスタンスを保持したままにしてしまいます。

```python
# multiprocessを閉じる
p.close()
p.join()
```

#### 並列処理しない場合


```python
before = time.time()
# 処理の実行
result = list(map(wait_sec, [1, 5, 3]))
after = time.time()

print('it took {} sec'.format(after-before))
print(result)
```

    it took 9.039642572402954 sec
    [1, 25, 9]


### 1つの引数による並列処理の実行

`Pool.map()`を使用すると結果はリストが返り、すべての計算が終了してから結果が得られます。<br>`Pool.imap()`を使用すると結果はイテレータとして返り、リストの順番通りに計算が終了し次第、結果が得られます。

#### `Pool.map()`


```python
# 物理コアの数-1を使用する物理コアの数としてPoolに指定
p = Pool(processes=cpu_count()-1)

before = time.time()
# 処理の実行
result = p.map(wait_sec, [1, 5, 3])
after = time.time()

# multiprocessを閉じる
p.close()
p.join()

print('it took {} sec'.format(after-before))
print(result)
```

    it took 5.004767179489136 sec
    [1, 25, 9]


#### `Pool.imap()`


```python
# 物理コアの数-1を使用する物理コアの数としてPoolに指定
p = Pool(processes=cpu_count()-1)

before = time.time()
# 処理の実行
# 処理ごとに実行にかかった時間を表示
for i in p.imap(wait_sec, [1, 5, 3]):
# imap_unorderedは処理が終わり次第返す
# for i in p.imap_unordered(wait_sec, [1, 5, 3]):　
    print('{}: {} s ec'.format(i, time.time() - before))
result = p.imap(wait_sec, [1, 5, 3])
after = time.time()

# multiprocessを閉じる
p.close()
p.join()

print('it took {} sec'.format(after-before))
```

    1: 1.0062987804412842 s ec
    25: 5.008757829666138 s ec
    9: 5.009317636489868 s ec
    it took 5.013044595718384 sec



```python
# p.imapの戻り値はiterableなので，tqdmを使うことも可能
p = Pool(processes=cpu_count()-1)
results = list(tqdm(p.imap(wait_sec, [1, 5, 3]), total=3))
# multiprocessを閉じる
p.close()
p.join()

print(result)
```

    100%|██████████| 3/3 [00:05<00:00,  1.67s/it]

    <multiprocessing.pool.IMapIterator object at 0x7fe24c59ba50>


    


## 複数の引数による並列処理

複数の引数による並列処理の場合はイテレータを使用できるラッパー関数を使用します


```python
# 複数の引数を入れる場合はラッパー関数を作る
# ２つの引数をとる関数
def multiply(a, b):
    # a秒間処理を待機
    time.sleep(a)
    return a * b
# ラッパー関数の作成
def wrap_multiply(args):
    # *はunpack演算子
    return multiply(*args)

params1 = [1, 2, 3, 4]
params2 = [10, 30, 70, 20]

# zipでタプルのリストを作成 （イテレータを返す）
# イテレータには引数にタプルを渡す
job_args = list(zip(params1, params2))
# 結果の確認
print(job_args)

# p.imapの戻り値はiterableなので，tqdmを使うことも可能
p = Pool(processes=cpu_count()-1)
results = tqdm(p.imap(wrap_multiply, job_args), total=len(params1))
# multiprocessを閉じる
p.close()
p.join()

list(results)
```

    [(1, 10), (2, 30), (3, 70), (4, 20)]


    100%|██████████| 4/4 [00:06<00:00,  1.52s/it]





    [10, 60, 210, 80]


