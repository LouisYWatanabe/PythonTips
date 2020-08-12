# `*args`と`**kwargs`
## *args
- `*args`とすることで、引数をいくつでも設定することが可能になります。<br>tapuleです

ラッパー関数を作成するときに使用します。<br>何か関数`f`があったときに、その関数`f`を使用したラッパー関数`f'`を作ると、<br>その後の使用者はラッパー関数`f'`を使用して関数`f`の中身は知らない状態で処理をさくせいするばあいが多くあります。<br>並列処理を行う際には良く起こる現象です。



```python
def func1(a, b):
    return a + b
```


```python
def func2(*args):
    return args[0] + args[1]
```


```python
func1(1, 3)
```




    4




```python
func2(1, 3, 3, 4, 5)
```




    4



return_only_png関数を作成し、<br>リストの引数からpngファイル名のみをリストで返す関数を作成します。<br>`['image1.png', 'image2.jpg']`引数から`['image1.png']`を返す


```python
file_list = ['image1.png', 'image2.jpg']
file_list[0][-3:]
```




    'png'




```python
def return_only_png(file_list):
    return_file_list = []
    for file_name in file_list:
        # 後ろ4文字を取得して内容が'.png'だったらリストに追加
        if file_name[-4:] == '.png':
            return_file_list.append(file_name)
    return return_file_list
```


```python
file_list = ['image1.png', 'image2.jpg']
return_only_png(file_list)
```




    ['image1.png']



1つ1つの文字列を使用してそれをリストに変えそうとすると<br>*argsを引数に使用する必要があります。


```python
def return_only_png(*args):
    return_file_list = []
    for file_name in args:
        # 後ろ4文字を取得して内容が'.png'だったらリストに追加
        if file_name[-4:] == '.png':
            return_file_list.append(file_name)
    return return_file_list
```


```python
return_only_png('image1.png', 'image2.jpg', 'image3.png')
```




    ['image1.png', 'image3.png']



### *argsはtupleです


```python
#　不特定多数の引数を受け取れる*args
def return_only_png(*args):
    png_list = []

    for filename in args:
        if filename[-3:] == 'png':
            png_list.append(filename)
    # argsの中身はtuple
    print("args' data type is {}".format(type(args)))    
    return png_list

#　実行すると，.pngの要素だけがリストで帰ってくる
return_only_png('image1.png', 'image2.jpg', 'image3.png', 'image4.jpeg')
```

    args' data type is <class 'tuple'>





    ['image1.png', 'image3.png']



## **kwargs (keyword arguments)

`**kwargs`は`*args`のdictionary版です。<br>引数が多くなる場合はこれを使うときれいに書くことが出来ます


```python
def print_dict(**kwargs):
    print(kwargs)
```


```python
print_dict(a=1, b=2)
```

    {'a': 1, 'b': 2}



```python
def print_dict(**kwargs):
    # .get('key')でkeyと紐づくvalueを取得できます
    param1 = str(kwargs.get('param1'))
    # 'param2'の部分にデフォルトでvalue:'default_value'を設定
    param2 = str(kwargs.get('param2', 'default_value'))
    param3 = str(kwargs.get('param3'))
    
    print('param1 is {}'.format(param1))
    print('param2 is {}'.format(param2))
    print('param3 is {}'.format(param3))
```


```python
print_dict(param1=1, param2=4, param3=4, q=3, a=0)
```

    param1 is 1
    param2 is 4
    param3 is 4



```python
print_dict(param1='beautiful_value', param3=4, q=3, a=0)
```

    param1 is beautiful_value
    param2 is default_value
    param3 is 4


#### * と **の正体 (Unpacking operator)

`Unpacking operator`とは、listにパックされた内容を`unpack`した状態にすることができます。


```python
a = [1, 2, 3, 4]
print(a)
print(*a)
```

    [1, 2, 3, 4]
    1 2 3 4


***は長さが違うリストもunpackして１つの変数入れれば，１つのリストとして扱えます。<br>また** ****も同様にdictionaryのunpackができます。**


```python
# 長さが違うリストもunpackして１つの変数入れれば，１つのリストとして扱える
list_a = [1, 2, 3]
list_b = [4, 5, 6, 7, 8]
list_a_b = [*list_a, *list_b]
print(list_a_b)

# **も同様にdictionaryのunpackができる．
dict_a = {'one': 1, 'two': 2}
dict_b = {'three': 3, 'four': 4, 'five':5}
dict_a_b = {**dict_a, **dict_b}
print(dict_a_b)
```

    [1, 2, 3, 4, 5, 6, 7, 8]
    {'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5}


## *argsによるラッパー関数の定義

ラップすることにより、tupleのリストを簡単に処理できるようになります


```python
#　例えばこんな関数を
def add_extention(filename, idx=0, extension='.png'):
    return '{}_{}{}'.format(filename, idx, extension)
add_extention('image', idx=1, extension='.pmg')
```




    'image_1.pmg'



#### ラップします


```python
def wrap_add_extention(args):
    return add_extention(*args)

# ラップした関数の引数は1つなのでこのように使います。
arg_tuple = ('iamge', 1, '.png')
wrap_add_extention(arg_tuple)
```




    'iamge_1.png'




```python
# tupleリストの作成
arg_list = [('iamge', 1, '.png'), ('iamge_test', 6, '.png'),  ('sample', 3, '.jpeg')]
# リスト内包表記で処理することができます
[wrap_add_extention(arg) for arg in arg_list]
```




    ['iamge_1.png', 'iamge_test_6.png', 'sample_3.jpeg']


