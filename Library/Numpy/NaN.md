# NumpyにおけるNaNの扱い

## np.nanとnp.isnan()


```python
import numpy as np
```


```python
# 負の値のlogをとると"nan"が返る
# logにマイナスはないため
neg_val = -10
np.log(neg_val)
```

    /opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:4: RuntimeWarning: invalid value encountered in log
      after removing the cwd from sys.path.





    nan




```python
# タイプはfloatなので，floatについてのエラメッセージがでたらnp.nanを疑う
type(np.nan)
```




    float



`NaN`のタイプは`float`です。

## np.nan()


```python
# nanチェックにはnp.isnan()を使う
nan_val = np.log(neg_val)
np.isnan(nan_val)
```

    /opt/anaconda3/lib/python3.7/site-packages/ipykernel_launcher.py:2: RuntimeWarning: invalid value encountered in log
      





    True




```python
# is np.nan　や ==np.nanではダメ
print(nan_val is np.nan)
print(nan_val == np.nan)
```

    False
    False


`NaN`のチェックは`np.nan()`を使用してください
