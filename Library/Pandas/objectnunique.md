# 特定のカラムの型の出現個数の表示

```python
# 列ごとの型数の確認
app_train.dtypes.value_counts()
# 列ごとの型数の確認
app_train.dtypes.value_counts()
```
```
float16    61
int8       37
object     16
float32     4
int32       2
int16       2
dtype: int64
```
```python
# object型のカラム名ごとの出現個数の表示
app_train.select_dtypes('object').apply(pd.Series.nunique, axis=0)
```
```
NAME_CONTRACT_TYPE             2
CODE_GENDER                    3
FLAG_OWN_CAR                   2
FLAG_OWN_REALTY                2
NAME_TYPE_SUITE                7
NAME_INCOME_TYPE               8
NAME_EDUCATION_TYPE            5
NAME_FAMILY_STATUS             6
NAME_HOUSING_TYPE              6
OCCUPATION_TYPE               18
WEEKDAY_APPR_PROCESS_START     7
ORGANIZATION_TYPE             58
FONDKAPREMONT_MODE             4
HOUSETYPE_MODE                 3
WALLSMATERIAL_MODE             7
EMERGENCYSTATE_MODE            2
dtype: int64
```
