# ストップワード除去

```python
train['temp_list'].head()
```
```
0    [id, have, responded, if, i, were, going]
1                                  [sooo, sad]
2                               [bullying, me]
3                           [leave, me, alone]
4                                   [sons, of]
Name: temp_list, dtype: object
```

```python
import nltk
from nltk.corpus import stopwords
# ストップワード除去関数
def remove_stopword(x):
    return [y for y in x if y not in stopwords.words('english')]
# ストップワードの除去
train['temp_list'] = train['temp_list'].apply(lambda x:remove_stopword(x))

train['temp_list'].head()
```
```
0    [id, responded, going]
1               [sooo, sad]
2                [bullying]
3            [leave, alone]
4                    [sons]
Name: temp_list, dtype: object
```