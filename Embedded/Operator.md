# 演算子(Operator)


```python
1 + 1
```




    2




```python
3 - 1
```




    2




```python
2 * 2
```




    4




```python
2 / 5
```




    0.4




```python
2 ** 3
```




    8




```python
4 / 3
```




    1.3333333333333333




```python
# 商のみ取得
4 // 3
```




    1




```python
# 余りの取得
5 % 3 # 5÷ 3 = 1 余り 2
```




    2




```python
i = 0
# 「i = i + 1」と同じ
# 後述のloop処理でよく使う
i += 1
i
```




    1




```python
3 > 1
```




    True




```python
3 >= 3 # >(greater than)=(or equal to)
```




    True




```python
3 == 2
```




    False




```python
3 != 2 #!(not)=(equal)
```




    True




```python
'h' in 'hello'
```




    True




```python
'a' in ['a', 'b', 'c']
```




    True




```python
True and False
```




    False




```python
(4 > 2) or (1 > 3)
```




    True




```python
a = 1
(a == 1) or (b == 1) #b==1は評価されない
```




    True




```python
(a == 2) and (banana == 2)
```




    False




```python
a = [1, 2]
b = [1, 2]
# == は値の判定
print(a == b)
# is　は同一オブジェクトかの判定
print(a is b)
print(a is not b)
```

    True
    False
    True



```python
# メモリの表示
print(id(a))
print(id(b))
```

    94404086120304
    140215110607248



```python
# Noneは「なにも値がない」という値が入っているイメージ
a = None
# Noneかどうかの判定には'is None'を使う
a is None
```




    True




```python
# NoneはUndefined（定義されていないもの）とは違う
a
```


```python
something
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    <ipython-input-22-8bfdf8048bf9> in <module>
          1 # NoneはUndefinedとは違う
    ----> 2 something
    

    NameError: name 'something' is not defined

