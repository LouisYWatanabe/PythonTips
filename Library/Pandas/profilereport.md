# 各特徴量を個別に把握する

```python
# warningsを無視する
import warnings
warnings.filterwarnings('ignore')


import pandas_profiling as pdp  # pandas_profilingのインポート
display(pdp.__version__) # バージョン確認
pdp.ProfileReport(df)  # レポートの作成
```

### 書式

	df: データフレーム


### 例

```python
import pandas as pd
# warningsを無視する
import warnings
warnings.filterwarnings('ignore')

# トレーニングデータ、テストデータを読み込み
train = pd.read_csv('../input/titanic/train.csv')
test = pd.read_csv('../input/titanic/test.csv')

import pandas_profiling as pdp  # pandas_profilingのインポート
display(pdp.__version__)
pdp.ProfileReport(train)  # レポートの作成

# train.profile_report()
```



### 結果の味方

#### pdp.ProfileReport(df)を実行することで、

- Overview(概要) データの行（Number of observations）列（Number of variables）数、欠損値の割合（Missing cells (%)）、データ型（Variable types）の把握ができます。
- Variables(特徴量の情報)  記述統計やヒストグラム等が表示されます
- Correlations(相関)
- Sample(先頭5行)
が表示されます

##### Overview(概要)

Overview(概要)には、
- Datasetinfo: 特徴の数、データ数、欠損値の割合、データサイズ
- Variables types: 型の情報
- Warnings: 特徴量の中で欠損値が多いものや、値に0が多いものカテゴリ変数で値の種類が多いもの、2特徴間で相関が高いものが表示されます。

##### Variables(特徴量の情報)

Variables(特徴量の情報)には、全ての特徴量について、データのユニーク数や欠損値数等の概要やヒストグラムが表示されます。

右下の「Toggle details」をクリックすることにより、数値データの場合は、
- Statistics
- Hisgram
- CommonValues
- Extreme Values
が表示されます。
それぞれ表示される内容は次のようになります

###### Statistics
Statisticsには、`Quantile statistics`と`Descriptive statistics`が表示されます。
	- `Quantile statistics`: 分位点の情報で、「最小値」、「5%点」、「四分位点」等が表示されます。
	- `Descriptive statistics`: 記述統計で、「平均」や「分散」、「標準偏差」等が表示されます
###### Hisgram
ヒストグラムが表示されます。
###### CommonValues
出現頻度が多い順に、値が表示されます。
末尾には、欠損値の数が表示されます。
###### Extreme Values
最小値の5つと最大値の5つが表示されます。

##### Correlations(相関)
Correlations(相関)には、ピアソン相関係数とスピアマン相関係数による相関プロットが表示されます
##### Sample(先頭5行)
Sample(先頭5行)は、先頭の5行が表示されます

