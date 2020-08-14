# indexを更新

### .reset_index()で再度indexを割り振る


```python
df = pd.read_csv('tmdb_5000_movies.csv')
df = df[df['original_language'] == 'ja']
# indexはもとのまま
df.head(2)
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
      <th>97</th>
      <td>15000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>NaN</td>
      <td>315011</td>
      <td>[{"id": 1299, "name": "monster"}, {"id": 7671,...</td>
      <td>ja</td>
      <td>シン・ゴジラ</td>
      <td>From the mind behind Evangelion comes a hit la...</td>
      <td>9.476999</td>
      <td>[{"name": "Cine Bazar", "id": 5896}, {"name": ...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2016-07-29</td>
      <td>77000000</td>
      <td>120.0</td>
      <td>[{"iso_639_1": "it", "name": "Italiano"}, {"is...</td>
      <td>Released</td>
      <td>A god incarnate. A city doomed.</td>
      <td>Shin Godzilla</td>
      <td>6.5</td>
      <td>143</td>
    </tr>
    <tr>
      <th>1471</th>
      <td>41677699</td>
      <td>[{"id": 16, "name": "Animation"}, {"id": 10751...</td>
      <td>NaN</td>
      <td>12429</td>
      <td>[{"id": 456, "name": "mother"}, {"id": 1357, "...</td>
      <td>ja</td>
      <td>崖の上のポニョ</td>
      <td>The son of a sailor, 5-year old Sosuke lives a...</td>
      <td>39.586760</td>
      <td>[{"name": "Studio Ghibli", "id": 10342}, {"nam...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2008-07-19</td>
      <td>187479518</td>
      <td>100.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>Welcome To A World Where Anything Is Possible.</td>
      <td>Ponyo</td>
      <td>7.5</td>
      <td>926</td>
    </tr>
  </tbody>
</table>
</div>




```python
# 新しくindexを振り直す
df.reset_index().head(2)
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
      <th>index</th>
      <th>budget</th>
      <th>genres</th>
      <th>homepage</th>
      <th>id</th>
      <th>keywords</th>
      <th>original_language</th>
      <th>original_title</th>
      <th>overview</th>
      <th>popularity</th>
      <th>...</th>
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
      <td>97</td>
      <td>15000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>NaN</td>
      <td>315011</td>
      <td>[{"id": 1299, "name": "monster"}, {"id": 7671,...</td>
      <td>ja</td>
      <td>シン・ゴジラ</td>
      <td>From the mind behind Evangelion comes a hit la...</td>
      <td>9.476999</td>
      <td>...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2016-07-29</td>
      <td>77000000</td>
      <td>120.0</td>
      <td>[{"iso_639_1": "it", "name": "Italiano"}, {"is...</td>
      <td>Released</td>
      <td>A god incarnate. A city doomed.</td>
      <td>Shin Godzilla</td>
      <td>6.5</td>
      <td>143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1471</td>
      <td>41677699</td>
      <td>[{"id": 16, "name": "Animation"}, {"id": 10751...</td>
      <td>NaN</td>
      <td>12429</td>
      <td>[{"id": 456, "name": "mother"}, {"id": 1357, "...</td>
      <td>ja</td>
      <td>崖の上のポニョ</td>
      <td>The son of a sailor, 5-year old Sosuke lives a...</td>
      <td>39.586760</td>
      <td>...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2008-07-19</td>
      <td>187479518</td>
      <td>100.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>Welcome To A World Where Anything Is Possible.</td>
      <td>Ponyo</td>
      <td>7.5</td>
      <td>926</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 21 columns</p>
</div>




```python
# もとのindexはdropする
df.reset_index(drop=True).head(2)
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
      <td>15000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>NaN</td>
      <td>315011</td>
      <td>[{"id": 1299, "name": "monster"}, {"id": 7671,...</td>
      <td>ja</td>
      <td>シン・ゴジラ</td>
      <td>From the mind behind Evangelion comes a hit la...</td>
      <td>9.476999</td>
      <td>[{"name": "Cine Bazar", "id": 5896}, {"name": ...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2016-07-29</td>
      <td>77000000</td>
      <td>120.0</td>
      <td>[{"iso_639_1": "it", "name": "Italiano"}, {"is...</td>
      <td>Released</td>
      <td>A god incarnate. A city doomed.</td>
      <td>Shin Godzilla</td>
      <td>6.5</td>
      <td>143</td>
    </tr>
    <tr>
      <th>1</th>
      <td>41677699</td>
      <td>[{"id": 16, "name": "Animation"}, {"id": 10751...</td>
      <td>NaN</td>
      <td>12429</td>
      <td>[{"id": 456, "name": "mother"}, {"id": 1357, "...</td>
      <td>ja</td>
      <td>崖の上のポニョ</td>
      <td>The son of a sailor, 5-year old Sosuke lives a...</td>
      <td>39.586760</td>
      <td>[{"name": "Studio Ghibli", "id": 10342}, {"nam...</td>
      <td>[{"iso_3166_1": "JP", "name": "Japan"}]</td>
      <td>2008-07-19</td>
      <td>187479518</td>
      <td>100.0</td>
      <td>[{"iso_639_1": "ja", "name": "\u65e5\u672c\u8a...</td>
      <td>Released</td>
      <td>Welcome To A World Where Anything Is Possible.</td>
      <td>Ponyo</td>
      <td>7.5</td>
      <td>926</td>
    </tr>
  </tbody>
</table>
</div>




```python
#元のdfを更新する
df = pd.read_csv('tmdb_5000_movies.csv')
df = df[df['original_language'] == 'ja']
df.reset_index(inplace=True)
#もしくは再代入
df = pd.read_csv('tmdb_5000_movies.csv')
df = df[df['original_language'] == 'ja']
df = df.reset_index()
```

### .set_index('カラム名')


```python
df = pd.read_csv('tmdb_5000_movies.csv')
# idカラムがindexになる
df.set_index('id').head(2)
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
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>19995</th>
      <td>237000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://www.avatarmovie.com/</td>
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
      <th>285</th>
      <td>300000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://disney.go.com/disneypictures/pirates/</td>
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
  </tbody>
</table>
</div>
