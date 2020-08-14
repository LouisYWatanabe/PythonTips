
# それぞれの値にいくつのレコードがあるかを取得
## .value_counts()


```python
# それぞれの値にいくつのレコードがあるかを取得
df['original_language'].value_counts()
```




    en    4505
    fr      70
    es      32
    de      27
    zh      27
    hi      19
    ja      16
    it      14
    cn      12
    ko      11
    ru      11
    pt       9
    da       7
    sv       5
    fa       4
    nl       4
    th       3
    he       3
    ta       2
    ro       2
    cs       2
    id       2
    ar       2
    af       1
    el       1
    ps       1
    hu       1
    sl       1
    no       1
    is       1
    xx       1
    tr       1
    pl       1
    ky       1
    vi       1
    nb       1
    te       1
    Name: original_language, dtype: int64



## .sort_values()


```python
# デフォルトは昇順 (ascending)
df.sort_values('budget').head()
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
      <th>runtime_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2401</th>
      <td>0</td>
      <td>[{"id": 53, "name": "Thriller"}]</td>
      <td>NaN</td>
      <td>31932</td>
      <td>[]</td>
      <td>en</td>
      <td>City of Ghosts</td>
      <td>A con man (Dillon) travels to Cambodia (also o...</td>
      <td>2.644860</td>
      <td>[]</td>
      <td>...</td>
      <td>2002-01-01</td>
      <td>0</td>
      <td>116.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>City of Ghosts</td>
      <td>5.4</td>
      <td>18</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3187</th>
      <td>0</td>
      <td>[{"id": 35, "name": "Comedy"}, {"id": 10402, "...</td>
      <td>NaN</td>
      <td>12621</td>
      <td>[{"id": 2176, "name": "music teacher"}, {"id":...</td>
      <td>en</td>
      <td>Hamlet 2</td>
      <td>From the same people that brought you "Little ...</td>
      <td>5.293363</td>
      <td>[]</td>
      <td>...</td>
      <td>2008-01-21</td>
      <td>0</td>
      <td>92.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>One high school drama teacher is about to make...</td>
      <td>Hamlet 2</td>
      <td>6.1</td>
      <td>56</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3183</th>
      <td>0</td>
      <td>[{"id": 10749, "name": "Romance"}, {"id": 18, ...</td>
      <td>NaN</td>
      <td>14624</td>
      <td>[{"id": 1158, "name": "grandfather grandson re...</td>
      <td>en</td>
      <td>The Ultimate Gift</td>
      <td>When his wealthy grandfather finally dies, Jas...</td>
      <td>4.976268</td>
      <td>[{"name": "The Ultimate Gift LLC", "id": 3914}...</td>
      <td>...</td>
      <td>2006-10-20</td>
      <td>0</td>
      <td>114.0</td>
      <td>[{"iso_639_1": "es", "name": "Espa\u00f1ol"}, ...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>The Ultimate Gift</td>
      <td>6.8</td>
      <td>78</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3179</th>
      <td>0</td>
      <td>[{"id": 35, "name": "Comedy"}, {"id": 10749, "...</td>
      <td>NaN</td>
      <td>24621</td>
      <td>[{"id": 187056, "name": "woman director"}]</td>
      <td>en</td>
      <td>Chasing Papi</td>
      <td>Playboy Thomas Fuentes has so far been able to...</td>
      <td>1.118511</td>
      <td>[{"name": "Fox 2000 Pictures", "id": 711}]</td>
      <td>...</td>
      <td>2003-04-16</td>
      <td>6126237</td>
      <td>80.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>Three women. Three cities. Three times the tro...</td>
      <td>Chasing Papi</td>
      <td>6.3</td>
      <td>16</td>
      <td>False</td>
    </tr>
    <tr>
      <th>3178</th>
      <td>0</td>
      <td>[{"id": 18, "name": "Drama"}]</td>
      <td>http://www.foxsearchlight.com/thesavages</td>
      <td>8272</td>
      <td>[{"id": 494, "name": "father son relationship"...</td>
      <td>en</td>
      <td>The Savages</td>
      <td>A sister and brother face the realities of fam...</td>
      <td>5.663889</td>
      <td>[{"name": "Fox Searchlight Pictures", "id": 43...</td>
      <td>...</td>
      <td>2007-01-19</td>
      <td>0</td>
      <td>114.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>NaN</td>
      <td>The Savages</td>
      <td>6.8</td>
      <td>110</td>
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>




```python
# ascending=Trueで降順 (descending)
df.sort_values('budget', ascending=False).head()
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
      <th>runtime_nan</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17</th>
      <td>380000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 28, "...</td>
      <td>http://disney.go.com/pirates/index-on-stranger...</td>
      <td>1865</td>
      <td>[{"id": 658, "name": "sea"}, {"id": 1316, "nam...</td>
      <td>en</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>Captain Jack Sparrow crosses paths with a woma...</td>
      <td>135.413856</td>
      <td>[{"name": "Walt Disney Pictures", "id": 2}, {"...</td>
      <td>...</td>
      <td>2011-05-14</td>
      <td>1045713802</td>
      <td>136.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>Live Forever Or Die Trying.</td>
      <td>Pirates of the Caribbean: On Stranger Tides</td>
      <td>6.4</td>
      <td>4948</td>
      <td>False</td>
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
      <td>False</td>
    </tr>
    <tr>
      <th>7</th>
      <td>280000000</td>
      <td>[{"id": 28, "name": "Action"}, {"id": 12, "nam...</td>
      <td>http://marvel.com/movies/movie/193/avengers_ag...</td>
      <td>99861</td>
      <td>[{"id": 8828, "name": "marvel comic"}, {"id": ...</td>
      <td>en</td>
      <td>Avengers: Age of Ultron</td>
      <td>When Tony Stark tries to jumpstart a dormant p...</td>
      <td>134.279229</td>
      <td>[{"name": "Marvel Studios", "id": 420}, {"name...</td>
      <td>...</td>
      <td>2015-04-22</td>
      <td>1405403694</td>
      <td>141.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}]</td>
      <td>Released</td>
      <td>A New Age Has Come.</td>
      <td>Avengers: Age of Ultron</td>
      <td>7.3</td>
      <td>6767</td>
      <td>False</td>
    </tr>
    <tr>
      <th>10</th>
      <td>270000000</td>
      <td>[{"id": 12, "name": "Adventure"}, {"id": 14, "...</td>
      <td>http://www.superman.com</td>
      <td>1452</td>
      <td>[{"id": 83, "name": "saving the world"}, {"id"...</td>
      <td>en</td>
      <td>Superman Returns</td>
      <td>Superman returns to discover his 5-year absenc...</td>
      <td>57.925623</td>
      <td>[{"name": "DC Comics", "id": 429}, {"name": "L...</td>
      <td>...</td>
      <td>2006-06-28</td>
      <td>391081192</td>
      <td>154.0</td>
      <td>[{"iso_639_1": "en", "name": "English"}, {"iso...</td>
      <td>Released</td>
      <td>NaN</td>
      <td>Superman Returns</td>
      <td>5.4</td>
      <td>1400</td>
      <td>False</td>
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
      <td>False</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 21 columns</p>
</div>
