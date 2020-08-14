
# DataFrameのiteration

## .apply()

ataFrameの各行に関数をapplyできます。

```python
# DataFrameの各行に関数をapplyする
# 'budget'が0なら'budget'にNaN，0以外ならその値を返す関数
def return_nan_if_zero(budget):
    if budget == 0:
        return np.nan
    else:
        return budget
    
return_nan_if_zero(0)
```




    nan




```python
# budgetカラムのSeriesにapplyすれば，各budgetに対して関数を実行した値がSeriesで返ってくる
df_movies['budget'].apply(return_nan_if_zero)
```




    0       237000000.0
    1       300000000.0
    2       245000000.0
    3       250000000.0
    4       260000000.0
               ...     
    4798       220000.0
    4799         9000.0
    4800            NaN
    4801            NaN
    4802            NaN
    Name: budget, Length: 4803, dtype: float64




```python
# それをもとのDataFrameにいれれば，カラムの値を更新できる
df_movies['budget'] = df_movies['budget'].apply(return_nan_if_zero)
df_movies.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>production_countries</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000.0</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2009-12-10</td>
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000.0</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000.0</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>[{"iso_3166_1": "GB", "name": "United Kingdom"...</td>
      <td>2015-10-26</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
    </tr>
    <tr>
      <th>3</th>
      <td>250000000.0</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 80, "nam...</td>
      <td>http://www.thedarkknightrises.com/</td>
      <td>49026</td>
      <td>[{"id": 849, "name": "dc comics"}, {"id": 853,...</td>
      <td>en</td>
      <td>The Dark Knight Rises</td>
      <td>Following the death of District Attorney Harve...</td>
      <td>112.312950</td>
      <td>[{"name": "Legendary Pictures", "id": 923}, {"...</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2012-07-16</td>
      <td>1084939099</td>
      <td>165.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>The Legend Ends</td>
      <td>The Dark Knight Rises</td>
      <td>7.6</td>
      <td>9106</td>
    </tr>
    <tr>
      <th>4</th>
      <td>260000000.0</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://movies.disney.com/john-carter</td>
      <td>49529</td>
      <td>[{"id": 818, "name": "based on novel"}, {"id":...</td>
      <td>en</td>
      <td>John Carter</td>
      <td>John Carter is a war-weary, former military ca...</td>
      <td>43.926995</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}]</td>
      <td>[{"iso_3166_1": "US", "name": "United States o...</td>
      <td>2012-03-07</td>
      <td>284139100</td>
      <td>132.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Lost in our world, found in another.</td>
      <td>John Carter</td>
      <td>6.1</td>
      <td>2124</td>
    </tr>
  </tbody>
</table>
</div>



### .apply(lambda x: )


```python
# applyに指定する関数をlambda関数にする
# まずはreturn_nan_if_zero()をlambda関数にして書いてみる
f = lambda x: np.nan if x==0 else x
f(0)
```




    nan




```python
df_movies = pd.read_csv('tmdb_5000_movies.csv')
# budgetとrevenueの0の値をnp.nanにする
df_movies['budget'] = df_movies['budget'].apply(lambda x: np.nan if x==0 else x)
df_movies['revenue'] = df_movies['revenue'].apply(lambda x: np.nan if x==0 else x)
# nanをdrop
df_movies = df_movies.dropna(subset=['revenue', 'budget'])
```

### .apply(lambda row: , axis=1)


```python
#行全体を使う
# revenue - budgetを別カラムで保持したいとする
df_movies['profit'] = df_movies.apply(lambda row: row['revenue'] - row['budget'], axis=1)
```


```python
# 最大のprofitをあげたmovieは？
df_movies.iloc[df_movies['profit'].idxmax()]
```




    budget                                                           2.37e+08
    genres                  [{"id": 28, "name": "Action"}, {"id": 12, "nam...
    homepage                                      http://www.avatarmovie.com/
    id                                                                  19995
    keywords                [{"id": 1463, "name": "culture clash"}, {"id":...
    original_language                                                      en
    original_title                                                     Avatar
    overview                In the 22nd century, a paraplegic Marine is di...
    popularity                                                        150.438
    production_companies    [{"name": "Ingenious Film Partners", "id": 289...
    production_countries    [{"iso_3166_1": "US", "name": "United States o...
    release_date                                                   2009-12-10
    revenue                                                       2.78797e+09
    runtime                                                               162
    spoken_languages        [{"iso_639_1": "en", "name": "English"}, {"iso...
    status                                                           Released
    tagline                                       Enter the World of Pandora.
    title                                                              Avatar
    vote_average                                                          7.2
    vote_count                                                          11800
    profit                                                        2.55097e+09
    Name: 0, dtype: object




```python
#フィルタを使っても同様に取得可能
df_movies[df_movies['profit'] == df_movies['profit'].max()]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>production_companies</th>
      <th>...</th>
      <th>release_date</th>
      <th>revenue</th>
      <th>runtime</th>
      <th>spoken_languages</th>
      <th>status</th>
      <th>tagline</th>
      <th>title</th>
      <th>vote_average</th>
      <th>vote_count</th>
      <th>profit</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>237000000.0</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
      <td>19995</td>
      <td>[{"id": 1463, "name": "culture clash"}, {"id":...</td>
      <td>en</td>
      <td>Avatar</td>
      <td>In the 22nd century, a paraplegic Marine is di...</td>
      <td>150.437577</td>
      <td>[{"name": "Ingenious Film Partners", "id": 289...</td>
      <td>...</td>
      <td>2009-12-10</td>
      <td>2.787965e+09</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>2.550965e+09</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 21 columns</p>
</div>



### .iterrows()


```python
# rowをイテレーションさせる イテレータは(idx, row)を返す
for idx, row in df_movies.iterrows():
    if row['vote_average'] == 10:
        print('{} got the higest score!!'.format(row['title']))
        print('vote counts: {}'.format(row['vote_count']))
        
# for _, row in df_movies.iterrows():        idx使わない場合は'＿'
```
