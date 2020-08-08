# scikit-learn

> 機械学習用モジュール

### 学習の流れ

1. 必要なモジュールのインポート
2. 学習させるデータの読み込み
3. データを「説明変数の訓練データ」、「目的変数の訓練データ」、「説明変数のテストデータ」、「目的変数のテストデータ」の4データに分割する
4. 学習器（学習モデル）の構築
5. 教師データ（訓練データ）を使い学習器で学習する


- <b>教師有り学習</b>
  - [回帰](./scikit-learn/回帰.md)
    	- [単回帰](./scikit-learn/線形単回帰.md)
    	- [重回帰](./scikit-learn/線形重回帰.md)
  - [分類](./scikit-learn/分類.md)
  	- [ロジスティック回帰](./scikit-learn/ロジスティック回帰.md)
  	- [線形SVM](./scikit-learn/線形SVM.md)
  	- [非線形SVM](./scikit-learn/非線形SVM.md)
	- [決定木](./scikit-learn/DecisionTreeClassifier/DecisionTreeClassifier.md)
  	- [決定木2](./scikit-learn/決定木.md)
		- [バギング](./scikit-learn/BaggingClassifier/BaggingClassifier.md)
			- [ランダムフォレスト](./scikit-learn/RandomForestClassifier/RandomForestClassifier.md)
  			- [ランダムフォレスト2](./scikit-learn/ランダムフォレスト.md)
	  	- **ブースティング**
			- [AdaBoost](./scikit-learn/AdaBoostClassifier/AdaBoostClassifier.md)
			- [勾配ブースティング](./scikit-learn/GradientBoostingClassifier/GradientBoostingClassifier.md)
		- [スタッキング](./scikit-learn/Stacking/Stacking.md)
  	- [K-近傍法](./scikit-learn/K-NN.md)
	- [**特徴量の重要度のグラフ**](./scikit-learn/feature_importance/feature_importance.md)
- **深層学習**
	- **CNNを用いた画像認識**
		- [CNNの流れと実装](./Library/scikit-learn/CNN1/CNN1.md)
		- [CNNを用いた画像認識の応用と実装](./Library/scikit-learn/CNN2/CNN2.md)
- <b>教師無し学習</b>
	- [k-means法](./scikit-learn/k-means.md)
	- [t-SNE](./scikit-learn/t-sne.md)

	- [主成分分析](./scikit-learn/PCA.md)
		- [前処理としての主成分分析](./scikit-learn/beforePCA.md)
	- <b>特徴量エンジニアリング</b>
		- [カテゴリ変数(名義尺度・順序尺度)の<br>エンコード(数値化)方法 ～順序のマッピング、LabelEncoderとOne Hot Encoder～](./scikit-learn/LabelEncoderOneHotEncoder.md)
		- [標準化（standardization）](./scikit-learn/standardization.md)
- [交差検証](./scikit-learn/交差検証.md)
- [正解率の取得](./scikit-learn/正解率.md)
- [適合率の取得](./scikit-learn/適合率.md)
- [再現率の取得](./scikit-learn/再現率.md)
- [F値の取得](./scikit-learn/F値.md)
- [混同行列](./scikit-learn/混同行列.md)
- [log loss](./scikit-learn/logloss.md)

- [複数のモデルの正解率の取得](./scikit-learn/複数のモデル.md)

[戻る](../Overallview.md)



```python
# 必要なモジュールのインポート
import request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
# 適合率、再現率、F値
from sklearn.metrics import precision_score, recall_score, f1_score

# 次に学習させたいデータの読み込む
# 以下のようにtrain_X, test_X, train_y, test_yという4つに分ける
train_X, test_X, train_y, test_y = make_regression(データの情報)

# 学習器の構築
model = LinearRegression()
# 教師データを使い学習
model.fit(train_X, train_y)
# 学習器の性能を確認するため決定係数を算出
score = model.score(test_X, test_y)

# 教師データとは別に用意したテスト用データを使い学習器に予測
pred_y = model.predict(test_X)

# test_yには正解のラベルを、y_predには予測結果のラベルをそれぞれ渡します
# 適合率
print("Precision: %.3f" % precision_score(test_y, y_pred))
# 再現率
print("Recall: %.3f" % recall_score(test_y, y_pred))
# F1
print("F1: %.3f" % f1_score(test_y, y_pred))

```