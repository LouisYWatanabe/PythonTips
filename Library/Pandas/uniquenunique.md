# ユニークな値と数


## .unique()と.nunique()


```python
# ユニークな値のみを取得
df_movies = pd.read_csv('tmdb_5000_movies.csv')
df_movies['original_language'].unique()
```




    array(['en', 'ja', 'fr', 'zh', 'es', 'de', 'hi', 'ru', 'ko', 'te', 'cn',
           'it', 'nl', 'ta', 'sv', 'th', 'da', 'xx', 'hu', 'cs', 'pt', 'is',
           'tr', 'nb', 'af', 'pl', 'he', 'ar', 'vi', 'ky', 'id', 'ro', 'fa',
           'no', 'sl', 'ps', 'el'], dtype=object)




```python
# ユニークな値の数を取得
df_movies['original_language'].nunique()
```




    37




```python
# idが本当にIDになっているのか
print(len(df_movies) == df_movies['id'].nunique())
```

    True

