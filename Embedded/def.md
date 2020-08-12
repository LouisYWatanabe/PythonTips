# 関数 (Function)


```python
def function_name(param1, param2):
    """
    This function prints something
    
    Parameters
    -------------------
        param1: explain param1
        param2: explain param2
    """
    print('Do something for {} and {}'.format(param1, param2))
    
    return param1 + ' and ' + param2

arg1 = 'argument1'
arg2 = 'argument2'
# 関数をcallするときは'関数名(引数)'
# 戻り値がある場合は変数に結果を格納
func_return = function_name(arg1, arg2)
func_return
```

    Do something for argument1 and argument2





    'argument1 and argument2'



## 返り値を変数に入れ忘れた時


```python
# 戻り値を変数に入れ忘れた場合は
function_name(arg1, arg2)
print('something')
```

    Do something for argument1 and argument2
    something



```python
# '_'に最後に実行した戻り値が入っています
_
```




    'argument1 and argument2'




```python
# 後から代入できます
value = _
value
```




    'argument1 and argument2'




```python
# print関数の返り値はNone
return_value = print('print')
return_value is None
```

    print





    True




```python
#引数が多い場合はfunction(param=arg)の形をとります
func_return = function_name(param1=arg1, param2=arg2)
```

    Do something for argument1 and argument2



```python

```
