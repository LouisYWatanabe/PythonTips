# csvの書き出し

```python
import pandas as pd

# 基本
df.to_csv('file_name.csv')

# index不要のとき (kaggle submission fileは不要なので忘れがち)
submission.to_csv('submission.csv', index=False)

# csvの書き出し
pd.to_csv('out/put/path/output.csv', encoding = 'shift-jis')
```

### 書式

	df: データフレーム

| path_or_buf | 出力するファイル名。省略した場合は、コンソール上に文字列として出力されます。                                                                                      |
|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| sep         | 区切り文字 (デフォルト: , (カンマ) )                                                                                                                              |
| index       | 行名を出力するかどうか。Falseを指定した場合、行名は出力されません。(デフォルト: True)                                                                             |
| encoding    | 出力する際の文字コード。’utf-8′, ‘shift_jis’, ‘euc_jp’ などを指定。参考: Python の文字コードの一覧 (デフォルト: Python 3 の場合 “utf-8”, Python 2 の場合 “ascii”) |

### 例

```python
import pandas as pd

sample4_2 = pd.read_csv("sample4.csv", header=None, names=["id","class","grade","name"])

sample4_2.to_csv('out/put/path/output.csv', encoding = 'shift-jis')
```

```python
    id class  grade       name
0    1     A      1      Satou
1    3     B      1  Hashimoto
2   15     B      3  Takahashi
3  102     A      2     Aikawa
```

```python
# profitの高い順に並べた表を保存
# 1. csv読み込み
df = pd.read_csv('tmdb_5000_movies.csv')
# 2. budgetとrevenueが0のものをフィルタアウト
df = df[~((df['revenue'] == 0) | (df['budget'] == 0))]
# 3. profit項目作成(revenue - budget)
df['profit'] = df.apply(lambda row: row['revenue'] - row['budget'], axis=1)
```


```python
df.head()
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
      <td>237000000</td>
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
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>2550965087</td>
    </tr>
    <tr>
      <th>1</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
      <td>285</td>
      <td>[{"id": 270, "name": "ocean"}, {"id": 726, "na...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>Captain Barbossa, long believed to be dead, ha...</td>
      <td>139.082615</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>...</td>
      <td>2007-05-19</td>
      <td>961000000</td>
      <td>169.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>At the end of the world, the adventure begins.</td>
      <td>Pirates of the Caribbean: At World's End</td>
      <td>6.9</td>
      <td>4500</td>
      <td>661000000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>245000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.sonypictures.com/movies/spectre/</td>
      <td>206647</td>
      <td>[{"id": 470, "name": "spy"}, {"id": 818, "name...</td>
      <td>en</td>
      <td>Spectre</td>
      <td>A cryptic message from Bond’s past sends him o...</td>
      <td>107.376788</td>
      <td>[{"name": "Columbia Pictures", "id": 5}, {"nam...</td>
      <td>...</td>
      <td>2015-10-26</td>
      <td>880674609</td>
      <td>148.0</td>
      <td>[{"iso_639_1": "fr", "name": "Fran\u00e7ais"},...</td>
      <td>Released</td>
      <td>A Plan No One Escapes</td>
      <td>Spectre</td>
      <td>6.3</td>
      <td>4466</td>
      <td>635674609</td>
    </tr>
    <tr>
      <th>3</th>
      <td>250000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 80, "nam...</td>
      <td>http://www.thedarkknightrises.com/</td>
      <td>49026</td>
      <td>[{"id": 849, "name": "dc comics"}, {"id": 853,...</td>
      <td>en</td>
      <td>The Dark Knight Rises</td>
      <td>Following the death of District Attorney Harve...</td>
      <td>112.312950</td>
      <td>[{"name": "Legendary Pictures", "id": 923}, {"...</td>
      <td>...</td>
      <td>2012-07-16</td>
      <td>1084939099</td>
      <td>165.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>The Legend Ends</td>
      <td>The Dark Knight Rises</td>
      <td>7.6</td>
      <td>9106</td>
      <td>834939099</td>
    </tr>
    <tr>
      <th>4</th>
      <td>260000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://movies.disney.com/john-carter</td>
      <td>49529</td>
      <td>[{"id": 818, "name": "based on novel"}, {"id":...</td>
      <td>en</td>
      <td>John Carter</td>
      <td>John Carter is a war-weary, former military ca...</td>
      <td>43.926995</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}]</td>
      <td>...</td>
      <td>2012-03-07</td>
      <td>284139100</td>
      <td>132.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Lost in our world, found in another.</td>
      <td>John Carter</td>
      <td>6.1</td>
      <td>2124</td>
      <td>24139100</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
df = df.sort_values('profit', ascending=False)
df.head()
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
      <td>237000000</td>
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
      <td>2787965087</td>
      <td>162.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Enter the World of Pandora.</td>
      <td>Avatar</td>
      <td>7.2</td>
      <td>11800</td>
      <td>2550965087</td>
    </tr>
    <tr>
      <th>25</th>
      <td>200000000</td>
      <td>[{"id": 18, "name": "Drama"}, {"id": 10749, "n...</td>
      <td>http://www.titanicmovie.com</td>
      <td>597</td>
      <td>[{"id": 2580, "name": "shipwreck"}, {"id": 298...</td>
      <td>en</td>
      <td>Titanic</td>
      <td>84 years later, a 101-year-old woman named Ros...</td>
      <td>100.025899</td>
      <td>[{"name": "Paramount Pictures", "id": 4}, {"na...</td>
      <td>...</td>
      <td>1997-11-18</td>
      <td>1845034188</td>
      <td>194.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Nothing on Earth could come between them.</td>
      <td>Titanic</td>
      <td>7.5</td>
      <td>7562</td>
      <td>1645034188</td>
    </tr>
    <tr>
      <th>28</th>
      <td>150000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.jurassicworld.com/</td>
      <td>135397</td>
      <td>[{"id": 1299, "name": "monster"}, {"id": 1718,...</td>
      <td>en</td>
      <td>Jurassic World</td>
      <td>Twenty-two years after the events of Jurassic ...</td>
      <td>418.708552</td>
      <td>[{"name": "Universal Studios", "id": 13}, {"na...</td>
      <td>...</td>
      <td>2015-06-09</td>
      <td>1513528810</td>
      <td>124.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>The park is open.</td>
      <td>Jurassic World</td>
      <td>6.5</td>
      <td>8662</td>
      <td>1363528810</td>
    </tr>
    <tr>
      <th>44</th>
      <td>190000000</td>
      <td>[{"id": 28, "name": "Action"}]</td>
      <td>http://www.furious7.com/</td>
      <td>168259</td>
      <td>[{"id": 830, "name": "car race"}, {"id": 3428,...</td>
      <td>en</td>
      <td>Furious 7</td>
      <td>Deckard Shaw seeks revenge against Dominic Tor...</td>
      <td>102.322217</td>
      <td>[{"name": "Universal Pictures", "id": 33}, {"n...</td>
      <td>...</td>
      <td>2015-04-01</td>
      <td>1506249360</td>
      <td>137.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Vengeance Hits Home</td>
      <td>Furious 7</td>
      <td>7.3</td>
      <td>4176</td>
      <td>1316249360</td>
    </tr>
    <tr>
      <th>16</th>
      <td>220000000</td>
      <td>[{"id": 878, "name": "Science Fiction"}, {"id"...</td>
      <td>http://marvel.com/avengers_movie/</td>
      <td>24428</td>
      <td>[{"id": 242, "name": "new york"}, {"id": 5539,...</td>
      <td>en</td>
      <td>The Avengers</td>
      <td>When an unexpected enemy emerges and threatens...</td>
      <td>144.448633</td>
      <td>[{"name": "Paramount Pictures", "id": 4}, {"na...</td>
      <td>...</td>
      <td>2012-04-25</td>
      <td>1519557910</td>
      <td>143.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Some assembly required.</td>
      <td>The Avengers</td>
      <td>7.4</td>
      <td>11776</td>
      <td>1299557910</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# csvファイルで保存 index=Falseでindexをカラムとして保存しない
df.to_csv('tmdb_5000_movies_profit_sorted.csv', index=False)
```
