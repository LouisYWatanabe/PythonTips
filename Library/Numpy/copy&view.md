# copy

```python
import numpy as np

arr_NumPy_copy = arr_NumPy[:].copy()
arr_NumPy_copy[0] = 100

```

### 書式

	arr[:].copy()

### 引数


### 例

```python
import numpy as np

# Pythonのリストでスライスを用いた場合の挙動を確認しましょう
arr_List = [x for x in range(10)]
print("Pythonのリスト型データです")
print("arr_List:",arr_List)
print() # 空行を出力

arr_List_copy = arr_List[:]
arr_List_copy[0] = 100

# Pythonのリストのスライスではコピーが作られるので、arr_Listにはarr_List_copyの変更が反映されません
print("Pythonのリストのスライスではコピーが作られるので、arr_Listにはarr_List_copyの変更が反映されません。")
print("arr_List:",arr_List)
print("arr_List_copy:",arr_List_copy)
print() # 空行を出力
print('-----------------------------------------------------------------------------------')

# NumPyのndarray配列でスライスを用いた場合の挙動を確認しましょう
arr_NumPy = np.arange(10)
print("NumPyのndarray配列データです")
print("arr_NumPy:",arr_NumPy)
print() # 空行を出力

arr_NumPy_view = arr_NumPy[:]
arr_NumPy_view[0] = 100

# NumPyのスライスではビューによって変数にデータが代入されるので、arr_NumPy_viewの変更がarr_NumPyに反映されます
# この時、変数名はデータが格納されている場所の情報を示しています
print("NumPyのスライスではビューによって変数にデータが代入されるので、arr_NumPy_viewの変更がarr_NumPyに反映されます。")
print("arr_NumPy:",arr_NumPy)
print("arr_NumPy_view:",arr_NumPy_view)
print() # 空行を出力
print('-----------------------------------------------------------------------------------')

# NumPyのndarray配列でcopy()を用いた場合の挙動を確認しましょう
arr_NumPy = np.arange(10)
print('NumPyのndarray配列でcopy()を用いた場合の挙動です')
print("arr_NumPy:",arr_NumPy)
print() # 空行を出力

arr_NumPy_copy = arr_NumPy[:].copy()
arr_NumPy_copy[0] = 100

# NumPyでもcopy()を用いた場合はコピーが作られるので、arr_Numpyにはarr_NumPy_copyの変更が反映されません。
print("NumPyでもcopy()を用いた場合はコピーが作られるので、arr_Numpyにはarr_NumPy_copyの変更が反映されません。")
print("arr_NumPy:",arr_NumPy)
print("arr_NumPy_copy:",arr_NumPy_copy)
```
```python

Pythonのリスト型データです
arr_List: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Pythonのリストのスライスではコピーが作られるので、arr_Listにはarr_List_copyの変更が反映されません。
arr_List: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
arr_List_copy: [100, 1, 2, 3, 4, 5, 6, 7, 8, 9]

-----------------------------------------------------------------------------------
NumPyのndarray配列データです
arr_NumPy: [0 1 2 3 4 5 6 7 8 9]

NumPyのスライスではビューによって変数にデータが代入されるので、arr_NumPy_viewの変更がarr_NumPyに反映されます。
arr_NumPy: [100   1   2   3   4   5   6   7   8   9]
arr_NumPy_view: [100   1   2   3   4   5   6   7   8   9]

-----------------------------------------------------------------------------------
NumPyのndarray配列でcopy()を用いた場合の挙動です
arr_NumPy: [0 1 2 3 4 5 6 7 8 9]

NumPyでもcopy()を用いた場合はコピーが作られるので、arr_Numpyにはarr_NumPy_copyの変更が反映されません。
arr_NumPy: [0 1 2 3 4 5 6 7 8 9]
arr_NumPy_copy: [100   1   2   3   4   5   6   7   8   9]
```

### 説明

スライスをコピーとして扱いたい場合にはarr[:].copy()を使用します。

