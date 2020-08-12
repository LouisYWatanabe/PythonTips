# 文字型

## Strings

シングルクォーテーションの中であれば、ダブルクォーテーションを使用することができます。<br>またダブルクォーテーションの中であれば、シングルクォーテーションを使用することができます。


```python
'single'
```




    'single'




```python
"double"
```




    'double'




```python
'''
関数の
レファレンス用の記述コード
'''
```




    '\n関数の\nレファレンス用の記述コード\n'



`\n`は改行を表します。


```python
# インデックスは0から始まります。
'hello'[0:3]
```




    'hel'




```python
# 文末から数えて4未満を表示
'file_ID.png'[:-4]
```




    'file_ID'




```python
'hrllo {}'.format('world')
```




    'hrllo world'




```python
# {}はカーリーブラケットと言います
'key is {K}, value is {v}'.format(K='KEYS', v='VALUES')
```




    'key is KEYS, value is VALUES'




```python
'hello' + ' ' + 'world'
```




    'hello world'




```python
patient_id = '1234'
number = '5'
file_name = patient_id + '_' + number + '.csv'
file_name
```




    '1234_5.csv'




```python
filename = '{}_{}.csv'.format(patient_id, number)
filename
```




    '1234_5.csv'



### 分割 split


```python
# split()
'hello world'.split( )    # 半角スペースで分割
```




    ['hello', 'world']




```python
'hello world'.split('o')
```




    ['hell', ' w', 'rld']




```python
filename.split('.')
```




    ['1234_5', 'csv']



### 結合 join

`'挿入文字列'.join()`：挿入文字列で複数の文字列を結合


```python
' '.join(['hello', 'world'])
```




    'hello world'




```python

```
