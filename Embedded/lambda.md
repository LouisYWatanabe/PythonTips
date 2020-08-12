# lambda関数

名前をつけるほどでもない一行で終わる関数に使う<br>ちょっとした計算をするときに使用します。


```python
# 例えばファイル名(拡張子除く)をPathから取ってくる関数
def get_filename(path):
    return path.split('/')[-1].split('.')[0]
```


```python
# .で分割した前半を抽出
'/home/user/Desktop/image1.jpg'.split('.')[0]
```




    '/home/user/Desktop/image1'




```python
# /で分割した末端を抽出
'/home/user/Desktop/image1.jpg'.split('/')[-1]
```




    'image1.jpg'



## lambda関数の書き方

- 関数でまず書きます
- `return`を削除し、1行にします
- def 関数名 → lambdaに変換します


```python
# lambda関数の書き方
lambda path: path.split('/')[-1].split('.')[0]
```




    <function __main__.<lambda>(path)>




```python
get_filename('/home/user/Desktop/image1.jpg')
```




    'image1'




```python
#lambda関数を変数に代入
x = lambda path: path.split('/')[-1].split('.')[0]
x('/home/user/Desktop/image1.jpg')
```




    'image1'




```python
def func(a):
    return a + 1
```


```python
# 上と同じ
x = lambda a: a + 1
```


```python
x(9)
```




    10



#### リストを引数にして，各要素に，'_<その要素のindex番号>.png'を付けたリス}トを返す関数


```python
def add_png(file_list):
    """
    リストを受け取り
    ファイル名_index番号.png
    をつけてリストで返す関数
    Parameters
        ilename_list: ファイル名のリスト
    """  
    return_list = []    # 返り値のリスト
    for idx, filename in enumerate(file_list):
        return_list.append('{}_{}{}'.format(filename, idx, '.png'))
    return return_list
```


```python
# 動作確認
filename_list = ['filename1', 'filename2', 'filename3']
add_png(filename_list)
```




    ['filename1_0.png', 'filename2_1.png', 'filename3_2.png']




```python
# リストの内包表記で関数を書く
def add_png(file_list):
    """
    リストを受け取り
    ファイル名_index番号.png
    をつけてリストで返す関数
    Parameters
        ilename_list: ファイル名のリスト
    """
    return ['{}_{}{}'.format(filename, idx, '.png') for idx, filename in enumerate(file_list)]
```


```python
# 動作確認
filename_list = ['filename1', 'filename2', 'filename3']
add_png(filename_list)
```




    ['filename1_0.png', 'filename2_1.png', 'filename3_2.png']




```python
# もっと汎用的なものにする
# .pngを引数にする
# リストの内包表記で関数を書く
def add_png(file_list, extension='.png'):
    """
    リストを受け取り
    ファイル名_index番号.png
    をつけてリストで返す関数
    Parameters
        ilename_list: list
            ファイル名のリスト
        extension: string
            拡張子(='.png')
    """  
    return ['{}_{}{}'.format(filename, idx, extension) for idx, filename in enumerate(file_list)]
```


```python
# 動作確認
filename_list = ['filename1', 'filename2', 'filename3']
add_png(filename_list)
```




    ['filename1_0.png', 'filename2_1.png', 'filename3_2.png']




```python
# '.jpg'でも同じ関数を使える
add_png(filename_list, '.jpg')
```




    ['filename1_0.jpg', 'filename2_1.jpg', 'filename3_2.jpg']



#### リストを引数にして，各要素に，'_<その要素のindex番号>.png'を付けたリストを返すlambda関数


```python
add_extention = lambda file_list, extension='.png': ['{}_{}{}'.format(filename, idx, extension) for idx, filename in enumerate(file_list)]
```


```python
# 動作確認
filename_list = ['filename1', 'filename2', 'filename3']
add_extention(filename_list)
```




    ['filename1_0.png', 'filename2_1.png', 'filename3_2.png']




```python
add_extention(filename_list, '.jpg')
```




    ['filename1_0.jpg', 'filename2_1.jpg', 'filename3_2.jpg']




```python

```
