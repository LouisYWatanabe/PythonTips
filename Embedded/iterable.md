# イテラブル(Iterable) と イテレータ(Iterator)

対象のデータ型がループ処理で使用できるかを判断するのに使用します。
イテラブルであれば`forloop`を使用できます。

- iterable: String, List, Tuple, Set, Dict
- not iterable: Integer, Float, Boolean

### Iterableとは，iter()関数でIteratorを返すオブジェクト
### Iteratorとは，next()関数でiterationできるオブジェクト


```python
colors = ['red', 'blue', 'green', 'yellow', 'white']  # リスト作成
for color in colors:
    print(color)
```

    red
    blue
    green
    yellow
    white


`colors`はリスト型でイテラブルなのでloop処理を行うことができました。


```python
for char in 'red':
    print(char)
```

    r
    e
    d



```python
# intはイテラブルではないのでエラーになります
for num in 40:
    print(num)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-3-9269cc267b25> in <module>
    ----> 1 for num in 40:
          2     print(num)


    TypeError: 'int' object is not iterable



```python
iter(colors)
```




    <list_iterator at 0x7f25786f9d50>



## iterとnext


```python
colors_i = iter(colors)
next(colors_i)
```




    'red'




```python
next(colors_i)
```




    'blue'



iterableは、1つ1つの要素を反復して取り出してます。<br>`for`文はこの処理を利用しています。


```python
r = range(10)
r_i = iter(r)
next(r_i)
```




    0




```python
next(r_i)
```




    1




```python
list(r)
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
# for loopで回せるもの→Iterableオブジェクト
# *ちなみにiteratorはiterableです
```


```python

```
