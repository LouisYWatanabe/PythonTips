# NumPy Arrayの保存とロード


```python
import numpy as np
```

## np.save('path', array)とnp.load('path')


```python
# n-dimentionalでも同じ
ndarray = np.random.randn(3, 4, 5)
```


```python
# numpyオブジェクトをsaveする
file_path = 'sample_ndarray.npy' #拡張子はつけなくても自動で.npyで保存される
np.save(file_path, ndarray)
```


```python
# numpyオブジェクトをloadする
loaded_ndarray = np.load(file_path)
```


```python
loaded_ndarray.shape
```




    (3, 4, 5)



### ndarrayに複数の情報を持たせて保存

一度dictionary型にして保存します。<br>
ndarrayに別の情報を付け加えて使用したときに使用します。


```python
# dictionaryを.npyとして保存する
dictionary = {
    'id': 123456,
    'image': np.array([1, 2, 3])
}
file_path = 'sample_dict.npy'
np.save(file_path, dictionary)
```


```python
# dictionaryはpickleで保存されているのでallow_pickle=Trueを指定してload
loaded_dict = np.load(file_path, allow_pickle=True)
```


```python
# arrayの状態で保存されているので，
loaded_dict
```




    array({'id': 123456, 'image': array([1, 2, 3])}, dtype=object)




```python
# dictionaryを取り出す場合は'[()]'を使う
loaded_dict[()]
```




    {'id': 123456, 'image': array([1, 2, 3])}


