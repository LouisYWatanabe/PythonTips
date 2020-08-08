# 欠損値確認関数

```python
# 欠損値計算関数
def missing_value_table(df):
    """欠損値の数とカラムごとの割合の取得
    Param : DataFrame
    確認を行うデータフレーム
    """
    # 欠損値の合計
    mis_val = df.isnull().sum()
    # カラムごとの欠損値の割合
    mis_val_percent = 100 * mis_val / len(df)
    # 欠損値の合計と割合をテーブルに結合
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    # カラム名の編集
    mis_val_table = mis_val_table.rename(
        columns={0:'Missing Values', 1:'% of Total Values'}
    )
    # データを欠損値のあるものだけにし。小数点以下1桁表示で降順ソートする
    mis_val_table = mis_val_table[mis_val_table.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False
    ).round(1)
    
    # 欠損値をもつカラム数の表示
    print('このデータフレームのカラム数は、', df.shape[1])
    print('このデータフレームの欠損値列数は、', mis_val_table.shape[0])
    
    # 欠損値データフレームを返す
    return mis_val_table
```
```python
# 欠損値情報の表示
Missing_value = missing_value_table(app_train)
Missing_value.head(20)
```
```
このデータフレームのカラム数は、 122
このデータフレームの欠損値列数は、 2
```

|                          |  Missing Values  | % of Total Values |
|-------------------------:|-----------------:|-------------------|
|          COMMONAREA_MEDI |           214865 |              69.9 |
|           COMMONAREA_AVG |           214865 |              69.9 |
|          COMMONAREA_MODE |           214865 |              69.9 |
| NONLIVINGAPARTMENTS_MEDI |           213514 |              69.4 |
| NONLIVINGAPARTMENTS_MODE |           213514 |              69.4 |
|  NONLIVINGAPARTMENTS_AVG |           213514 |              69.4 |
|       FONDKAPREMONT_MODE |           210295 |              68.4 |
|    LIVINGAPARTMENTS_MODE |           210199 |              68.4 |
|    LIVINGAPARTMENTS_MEDI |           210199 |              68.4 |
|     LIVINGAPARTMENTS_AVG |           210199 |              68.4 |
|           FLOORSMIN_MODE |           208642 |              67.8 |
|           FLOORSMIN_MEDI |           208642 |              67.8 |
|            FLOORSMIN_AVG |           208642 |              67.8 |
|         YEARS_BUILD_MODE |           204488 |              66.5 |
|         YEARS_BUILD_MEDI |           204488 |              66.5 |
|          YEARS_BUILD_AVG |           204488 |              66.5 |
|              OWN_CAR_AGE |           202929 |              66.0 |
|             LANDAREA_AVG |           182590 |              59.4 |
|            LANDAREA_MEDI |           182590 |              59.4 |
|            LANDAREA_MODE |           182590 |              59.4 |