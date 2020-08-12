# リスト型

list []：どんな方でも入れることは可能です。<br>しかし、基本的には同じ型のみを入れてください。


```python
lists = [1, 'hello', 3]
lists
```




    [1, 'hello', 3]




```python
lists[0]
```




    1



`.append()`を追加することでリストにデータを追加できます。


```python
lists2 = ['one', 'two']
lists2.append('three')

lists2
```




    ['one', 'two', 'three']




```python
lists2.append(lists)

lists2
```




    ['one', 'two', 'three', [1, 'hello', 3]]



リストの中の4つめの要素として`lists`の内容が入ります。<br>ただ、二重リストはNumpyを使用して作成することがほとんどです。


```python
lists2[3]
```




    [1, 'hello', 3]




```python
lists2[3][1]
```




    'hello'




```python
[1, 2, 3] + [4, 5, 6]
```




    [1, 2, 3, 4, 5, 6]




```python

```
