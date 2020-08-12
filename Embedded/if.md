# if文

```python
if (条件文1):
    (条件文1がTrueの場合の処理)
elif: (条件文2):
    (条件文1がFalseで条件文2がTrueの場合の処理)
else:
    (条件文1も条件文2もFalseの場合の処理)
```


```python
if 1 < 3:
    print('1 is less than 3')
```

    1 is less than 3



```python
if 1 == 2:
    print('1 is 2')
else:
    print('1 is not 2')
```

    1 is not 2



```python
if 1 == 2:
    print('1 is 2')
elif 1 != 2:
    print('1 is not 2')
else:
    print('something else')
```

    1 is not 2



```python
# if文を一行で書くことも可能．後述するlambda関数以外ではあまり使わない
# この場合はelseが必要なので注意
print('1 is 1') if 1 == 1 else print('1 is not 1')
```

    1 is 1


`(Trueの時の処理) if 1 == 1 (Falseの時の処理)`と書くような流れです。


```python

```
