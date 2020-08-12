# Loop

## for:リストの各要素に対しての処理を回す


```python
# リストの作成
colors = ['red', 'blue', 'green', 'yellow', 'white']
favorite_color = 'blue'   # フラグ

for color in colors:
    if color == favorite_color:
        print('{} is my favorite color'.format(color))
    else:
        print('{} is not my favorite color'.format(color))
```

    red is not my favorite color
    blue is my favorite color
    green is not my favorite color
    yellow is not my favorite color
    white is not my favorite color


## 連番リストを作る


```python
# 連番のリストを作りたいときはrange関数が便利
# range(start, stop, step) ==> [start, start+step, start+2*step.., stop未満]のリストを作れる
# startのデフォルトは0, step のデフォルトは１
list(range(10))
```




    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]




```python
ren = range(2, 10, 2)
ren
```




    range(2, 10, 2)




```python
list(range(2, 10, 2))
```




    [2, 4, 6, 8]




```python
# for文で回すときはlist()は不要
for i in range(10):
    print(i)
```

    0
    1
    2
    3
    4
    5
    6
    7
    8
    9


## リストのindexと要素を使いたいとき


```python
# リストの作成
colors = ['red', 'blue', 'green', 'yellow', 'white']
favorite_color = 'blue'   # フラグ

# リストのindexと要素どちらも使いたいときはenumerate()を使う
for idx, color in enumerate(colors):
    if color == favorite_color:
        print('{}: {} is my favorite color'.format(idx, color))
    else:
        print('{}: {} is not my favorite color'.format(idx, color))

```

    0: red is not my favorite color
    1: blue is my favorite color
    2: green is not my favorite color
    3: yellow is not my favorite color
    4: white is not my favorite color


### breakとcontinue


```python
# breakとcontinue
favorite_color = 'blue'
for idx, color in enumerate(colors):
    if color == favorite_color:
        continue
    else:
        print('{}: {} is not my favorite color...'.format(idx, color))
```

    0: red is not my favorite color...
    2: green is not my favorite color...
    3: yellow is not my favorite color...
    4: white is not my favorite color...



```python
# breakとcontinue
favorite_color = 'blue'
for idx, color in enumerate(colors):
    if color == favorite_color:
        break
    else:
        print('{}: {} is not my favorite color...'.format(idx, color))
```

    0: red is not my favorite color...


## リスト内包表記（List Comprehensions）


```python
colors = ['red', 'blue', 'green', 'yellow', 'white']

#例えば各要素に文字列の最後に'.png'をつけたい場合
# for loopを使うとこんな感じ
new_colors = []
for color in colors:
    new_color = color + '.png'
    new_colors.append(new_color)

new_colors
```




    ['red.png', 'blue.png', 'green.png', 'yellow.png', 'white.png']




```python
# リスト内包表記での書き方
[color + '.png' for color in colors]
```




    ['red.png', 'blue.png', 'green.png', 'yellow.png', 'white.png']



## while loop


```python
# 条件が満たされるまでのループ
i = 0
while i < 5:
    print('{} is less than 0'.format(i))    # 処理（条件文がTrueになるようにする）
    # これがないと無限ループになるので注意
    i += 1
```

    0 is less than 0
    1 is less than 0
    2 is less than 0
    3 is less than 0
    4 is less than 0



```python

```
