# mutableとimmutable

変更可能なオブジェクトと変更不可なオブジェクト

- mutable : List, Set, Dictinary
- immutable: Integer, Float, Boolean, String, Tuple


- mutable : 容量が大きくなりがちなオブジェクト
- immutable: 容量が大きくないオブジェクト

## mutable


```python
def append_elem(list_param, elem=0):
    
    list_param.append(elem)
    
    return list_param

list_a = [1, 2, 3]
list_b = append_elem(list_a, elem=4)
# list_bは当然 list_aにelemを追加したリスト
print('list_b : {}'.format(list_b))
# じゃぁlist_aは？
print('list_a : {}'.format(list_a))
```

    list_b : [1, 2, 3, 4]
    list_a : [1, 2, 3, 4]


引数の`list_a`も値の追加が反映されてしまします。<br>関数は引数のメモリ領域を渡す（参照渡しな）ので値が変更されてしまします。


```python
list_a = [1, 2, 3]
print(list_a)
print(id(list_a))
list_a += [4, 5, 6]
print(list_a)
print(id(list_a))
```

    [1, 2, 3]
    140170835094016
    [1, 2, 3, 4, 5, 6]
    140170835094016


一度確保したメモリ領域を変えずに値の変更を行っていることが分かります。

## immutable


```python
a = 3
print(id(a))
a += 4
print(a)
print(id(a))
```

    94702176600896
    7
    94702176601024


変数を変更した時にメモリ領域も変わっています。<br>immutableなオブジェクトです。


```python
list_a = [1, 2, 3]

def append_elem(list_a, elem):
    list_b = list_a.append(elem)
    return list_b

list_b = append_elem(list_a, elem=4)
list
```


```python
def add_num(param, num=0):
    
    param += num
    
    return param

num_a = 4
num_b = add_num(num_a, num=3)
# 当然num_bは
print(num_b)
# じゃぁ元のnum_aは？
print(num_a)
```

    7
    4


引数の`num_a`の値は変更されません。<br>関数は引数の値がimmutableの時、メモリの別領域に値を渡します。


```python
# mutableオブジェクト
a = ['hello']
print('original id: {}'.format(id(a)))
# 変更しても
a += ['world']
# idは変わっていない　もとのメモリは上書きされていない
print('updated id: {}'.format(id(a)))
print(a)
```

    original id: 140170835582000
    updated id: 140170835582000
    ['hello', 'world']



```python
# immutableオブジェクト
a = 'hello'
print('original id: {}'.format(id(a)))
# 変更すると
a += ' world'
# idが変わっている・・・！　もとのメモリは上書きされていない
print('updated id: {}'.format(id(a)))
print(a)
```

    original id: 140170835232048
    updated id: 140170835263600
    hello world



```python
# Pythonは基本参照渡し
def print_id(n):
    print(id(n))
    
a = 'test'
# 表示されるIDは同じ．つまり参照が渡されている
print(id(a))
print_id(a)
# しかし，immutableでは元のメモリを更新できないので，値渡しのような挙動をとる
```

    140170934939888
    140170934939888



```python

```
