# 辞書型

`key:value`がペアになって要素を作成しています。<br>`key`や`value`はどんな型でもよいです。


```python
dict1 = {'key1':'value1', 'key2':2, 3:'value3'}
dict1
```




    {'key1': 'value1', 'key2': 2, 3: 'value3'}




```python
dict1['key1']
```




    'value1'




```python
# 以下のように指定しても 0 というkeyはないのでエラーとなります
# dict1[0]
```


```python
dict1[3]
```




    'value3'



### 辞書型への追加


```python
dict1['key4'] = 'value4'
dict1
```




    {'key1': 'value1', 'key2': 2, 3: 'value3', 'key4': 'value4'}



### 既に`key`がある場合

`value`が上書きされます。


```python
dict1['key1'] = 'value uppdate'
dict1
```




    {'key1': 'value uppdate', 'key2': 2, 3: 'value3', 'key4': 'value4'}



### keyやvalueのリストの取得


```python
dict1.keys()
```




    dict_keys(['key1', 'key2', 3, 'key4'])




```python
dict1.values()
```




    dict_values(['value uppdate', 2, 'value3', 'value4'])




```python

```
