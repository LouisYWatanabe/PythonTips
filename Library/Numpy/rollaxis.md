
### np.rollaxis()

Numpy配列の順番を入れ替えます。


```python
a = np.zeros([4, 3, 2])
a.shape
```




    (4, 3, 2)




```python
# 1番目の形状を0番目と入れ替える
np.rollaxis(a, axis=1, start=0).shape
```




    (3, 4, 2)




```python
a = np.zeros([4, 3, 2, 1])
a.shape
```




    (4, 3, 2, 1)




```python
# 2番目の形状を3番目と入れ替える(4番目は存在しないので存在する最大数の3番目が適用される)
np.rollaxis(a, axis=2, start=4).shape
```




    (4, 3, 1, 2)


