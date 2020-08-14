# ライブラリ

- [**ライブラリ**](./Library.md)
  - [多クラス分類表示](./Library/classifier_plot.md)
  - [numpy](./Library/numpy.md)
	- <b>前処理</b>
		- [文字列を数値に変換(indexに変換)](./Library/Numpy/enumerate.md)
		- [配列をカウント](./Library/Numpy/bincount.md)
		- [重複無しの要素抽出](./Library/Numpy/unique.md)
		- [型の確認](./Library/Numpy/type.md)
		- [変数の種類の確認](./Library/Numpy/unique.md)
		- [変換 型](./Library/Numpy/astype.md)
		- [列の取得](./Library/Numpy/列の取得.md)
	- [一次元 多次元配列の作成・型宣言](./Library/Numpy/stack_arrays.md)
		- [数値のArrayを関数で作成(arange, linspace, logspace)](./Library/Numpy/arangelinspace.md)
	- [1次元配列の追加](./Library/Numpy/stack_array.md)
	- [1次元配列の計算](./Library/Numpy/math.md)
	- [行列の確認・reshape・次元操作 shape expand_dims squeeze flatten](./Library/Numpy/shape.md)
	- [行列の要素取得とスライシング](./Library/Numpy/IndexingSlicing.md)
	- [要素を指定して行列生成 (zeros, ones, eye)](./Library/Numpy/zerosones.md)
	- [ユニバーサル関数](./Library/Numpy/ansanble.md)
	- [配列同士の連結](./Library/Numpy/array.md)
	- [コピー](./Library/Numpy/copy&view.md)
	- [インデックス参照とスライス](./Library/Numpy/index.md)
	- [ブールインデックス参照](./Library/Numpy/blueindex.md)
	- [乱数 random](./Library/Numpy/rand.md)
	- [統計量 標準偏差最大・最小値 標準偏差](./Library/Numpy/Statistics.md)
	- [数学用関数 平方根 np.sqrt() 対数 np.log() 指数関数 np.exp() ネイピア数 np.e 合計 np.sum 絶対値 np.abs()](./Library/Numpy/Mathfunction.md)
	- [NumpyにおけるNaNの扱い np.nanとnp.isnan()](./Library/Numpy/NaN.md)
	- [ndarrayの中の要素の操作 np.clip(a, a_min, a_max) np.where() all() any() np.unique()](./Library/Numpy/ndarraycondition.md)
	- [結合と転置 np.concatenate() np.stack() np.transposeと.T ](./Library/Numpy/stackT.md)
	- [ndarrayの保存とロード list dictionary np.save('path', array) np.load('path')](./Library/Numpy/saveroad.md)

  - [pandas](./Library/pandas.md)
	- [データフレームのメモリ削減関数](./Library/Pandas/reduce_mem_usage.md)
	- [複数データの結合](./Library/Pandas/HomeCreditFeatureEngineering.md)

	- <b>pandas-profiling(使用する場合は、pip installまたはconda installが必要 )</b>
    	-  [各特徴量を個別に把握する](./Library/Pandas/profilereport.md)
	- <b>データ構造</b>
	  - [Series](./Library/Pandas/Series.md)
		- [Seriesの作成](./Library/Pandas/Seriesの作成.md)
		- [データ・インデックスの抽出（Series）](./Library/Pandas/data_indexget.md)
		- [要素の追加（Series）](./Library/Pandas/data_attend.md)
		- [要素の削除（Series）](./Library/Pandas/data_del.md)

	  - [DataFrame](./Library/Pandas/DataFrame.md)
		- [DataFrameの作成](./Library/Pandas/DataFrameの作成.md)
        - [csvの読み込み](./Library/Pandas/read_csv.md)
        - [csvの書き出し](./Library/Pandas/write_csv.md)
		- [インデックス・カラムの取得・設定 index columns](./Library/Pandas/data_indexset.md)
        - [カラムとローの省略拒否](./Library/Pandas/set_option.md)
		- [表結合 concat() merge()](./Library/Pandas/concatmerge.md)

            - [縦連結](./Library/Pandas/concat.md)
            - [結合 merge](./Library/Pandas/merge.md)
        
		- [ユニークな値と数 .unique() .nunique()](./Library/Pandas/uniquenunique.md)
		- [それぞれの値にいくつのレコードがあるか .value_counts() .sort_values()](./Library/Pandas/value_counts.md)

		- [DataFrameのiteration  .apply() .iterrows() ](./Library/Pandas/applyiterrows.md)
		- [ピボットテーブルの作り方 .pivot_table() .xs()](./Library/Pandas/pivot_tablexs.md)
		- [データの表示確認（リスト）](./Library/Pandas/hyouzi.md)
        - [欠損値の有無の確認](./Library/Pandas/isnull.md)
            - [**関数** 欠損値の有無の確認](./Library/Pandas/missing_value_table.md)
        - [欠損値の削除](./Library/Pandas/dropna.md)
        - [欠損値の補完](./Library/Pandas/fillna.md)
        - [列の削除](./Library/Pandas/drop.md)
        - [型の確認](./Library/Pandas/type.md)
        - [特定の値をNA（欠損値）に変換 replace](./Library/Pandas/replace.md)
        - [まとめて型変換 replace](./Library/Pandas/all_replace.md)
        - [外れ値の変換 clip](./Library/Pandas/clip.md)
        - [行と列数の確認](./Library/Pandas/shape.md)
        - [列の名前の確認](./Library/Pandas/colum.md)
        - [概要の確認](./Library/Pandas/info.md)
        - [要約統計量の表示](./Library/Pandas/describe.md)
        - [抽出・集計（特定の単語）](./Library/Pandas/gotyuu.md)
        - [特定の区切りでデータを分割 binning](./Library/Pandas/binning.md)
        - [要素の個数、頻度（出現回数）をカウント（Value_Counts()）](./Library/Pandas/valuecounts.md)
        - [カテゴリー変数を質的変数に指定した値で変換（map()）](./Library/Pandas/map.md)
        - [データから重複しないように人名を抽出](./Library/Pandas/データから重複しないように人名を抽出.md)
        - [特定のカラムの型の出現個数の表示](./Library/Pandas/objectnunique.md)
        - [Label encoding **and** One-hot encoding](./Library/Pandas/LabelEncoder.md)
            - [One-Hot表現 ダミー変数に変換](./Library/Pandas/one-hot.md)
        - [同じ値を持つデータをまとめて、それぞれの塊に対して共通の操作を行うgroupby](./Library/Pandas/groupby.md)
        - <b>自然言語処理</b>
            - [小文字を大文字に、大文字を小文字に変換(upper lower)](./Library/Pandas/upperlower.md)
            - [文字列の分割(split())](./Library/Pandas/split.md)
            - [積集合$A\bigcap B$(a.intersection(b))](./Library/Pandas/intersection.md)
            - [特定の文字列を分割して新特徴量にし、One-hot encoding](./Library/Pandas/PredictFutureSales.md)
		- [行の連結](./Library/Pandas/行の連結.md)
		- [DataFrameの連結](./Library/Pandas/連結.md)
		- [列の追加](./Library/Pandas/列の追加.md)
		- [結合（マージ）](./Library/Pandas/結合.md)
		- [同名でない列の結合（マージ）](./Library/Pandas/同名でない結合.md)
		- [インデックスをKeyにして結合](./Library/Pandas/インデックス結合.md)
		- [名前による参照](./Library/Pandas/名前による参照.md)
		- [番号による参照](./Library/Pandas/番号による参照.md)
		- [行または列の削除](./Library/Pandas/行または列の削除.md)
		- [一部の行の取得（.head(), .tail(), .column()）](./Library/Pandas/一部の行取得.md)
		- [要約統計量の取得 describe](./Library/Pandas/要約統計量.md)
		- [行間列間の差を取得 diff](./Library/Pandas/行間列間の差.md)
		- [グループ化(groupby)](./Library/Pandas/グループ化.md)
		- [ソート(sort_values)](./Library/Pandas/sort_values.md)
	    - [ソート](./Library/Pandas/ソート.md)
	    - [フィルタリング](./Library/Pandas/filterling.md)
	    - [indexを更新 再度indexを割り振る reset_index() set_index('カラム名')](./Library/Pandas/index.md)

	    - [参照](./Library/Pandas/参照.md)

  - [Matplotlib（日本語使用：pip install japanize-matplotlib）import japanize_matplotlib](./Library/Matplotlib.md)
    - [グラフの基本表示](./Library/Matplotlib/グラフの基本表示.md)

  - [seaboen](./Library/seaborn.md)
    - [集計表とクロス集計](./Library/Seaborn/集計表.md)
    - [ヒストグラム](./Library/Seaborn/hist.md)<br><br>

  - [LightGBMとoptunaと交差検証](./Library/optuna_lgb.md)
	- [LightGBMと複数データのEDA](./Library/PredictFutureSales/PredictFutureSales.md)

  - [scikit-learn](./Library/scikit-learn.md)
	- <b>教師有り学習</b>
		- [回帰](./Library/scikit-learn/回帰.md)
			- [単回帰](./Library/scikit-learn/線形単回帰.md)
			- [重回帰](./Library/scikit-learn/線形重回帰.md)
		- [分類](./Library/scikit-learn/分類.md)
			- [ロジスティック回帰](./Library/scikit-learn/ロジスティック回帰.md)
			- [線形SVM](./Library/scikit-learn/線形SVM.md)
			- [非線形SVM](./Library/scikit-learn/非線形SVM.md)
			- [決定木](./Library/scikit-learn/DecisionTreeClassifier/DecisionTreeClassifier.md)
			- [決定木2](./Library/scikit-learn/決定木.md)
				- [バギング](./Library/scikit-learn/BaggingClassifier/BaggingClassifier.md)
					- [ランダムフォレスト](./Library/scikit-learn/RandomForestClassifier/RandomForestClassifier.md)
					- [ランダムフォレスト2](./Library/scikit-learn/ランダムフォレスト.md)
				- **ブースティング**
					- [AdaBoost](./Library/scikit-learn/AdaBoostClassifier/AdaBoostClassifier.md)
					- [勾配ブースティング](./Library/scikit-learn/GradientBoostingClassifier/GradientBoostingClassifier.md)
					- [XGBClassifier](./Library/scikit-learn/xgboost_XGBClassifier.md)
				- [スタッキング](./Library/scikit-learn/Stacking/Stacking.md)
			- [K-近傍法](./Library/scikit-learn/K-NN.md)
			- [**特徴量の重要度のグラフ**](./Library/scikit-learn/feature_importance/feature_importance.md)
		- **深層学習**
			- **CNNを用いた画像認識**
				- [CNNの流れと実装](./Library/scikit-learn/CNN1/CNN1.md)
				- [CNNを用いた画像認識の応用と実装](./Library/scikit-learn/CNN2/CNN2.md)
	- <b>教師無し学習</b>
		- [k-means法](./Library/scikit-learn/k-means.md)
		- [t-SNE](./Library/scikit-learn/t-sne.md)
		- [主成分分析](./Library/scikit-learn/PCA.md)
			- [前処理としての主成分分析](./Library/scikit-learn/beforePCA.md)
	- <b>特徴量エンジニアリング</b>
		- [カテゴリ変数(名義尺度・順序尺度)の<br>エンコード(数値化)方法 ～順序のマッピング、LabelEncoderとOne Hot Encoder～](./Library/scikit-learn/LabelEncoderOneHotEncoder.md)
		- [標準化（standardization）](./Library/scikit-learn/standardization.md)


	- [交差検証](./Library/scikit-learn/交差検証.md)
	- [正解率の取得](./Library/scikit-learn/正解率.md)
	- [適合率の取得](./Library/scikit-learn/適合率.md)
	- [再現率の取得](./Library/scikit-learn/再現率.md)
	- [F値の取得](./Library/scikit-learn/F値.md)
	- [混同行列](./Library/scikit-learn/混同行列.md)
	- [log loss_cross entropy](./Library/scikit-learn/logloss.md)
	- [複数のモデルの正解率の取得](./Library/scikit-learn/複数のモデル.md)

  - [statsmodels](./Library/statsmodels.md)
	- [単回帰](./Library/statsmodels/Linear_regression.md)
  - [TensorFlow](./Library/tensorflow.md)
	- [TensorFlowの演算子](./Library/tensorflow/TensorFlowmath.md)
	- [tf.constant 定数の値を保持する](./Library/tensorflow/tf.constant.md)
	- [tf.Variable 変数の値を保持する](./Library/tensorflow/tf.variable.md)
	- [tf.truncated_normal 正規分布ランダムサンプリング](./Library/tensorflow/tf.truncatednormal.md)
	- [tf.placeholder データの保持](./Library/tensorflow/tf.placeholder.md)
	- [tf.Session Sessionオブジェクトの生成](./Library/tensorflow/tf.session.md)
	- [tf.run opノードを評価し、operationの結果を返す](./Library/tensorflow/tf.run.md)
	- [tf.train.GradientDescentOptimizer 勾配降下アルゴリズム](./Library/tensorflow/tf.train.GradientDescentOptimizer.md)
	- [tf.matmul 行列の掛算](./Library/tensorflow/tf.matmul.md)

  - [keras](./Library/keras.md)
	- [学習の流れ](./Library/keras/XOR.md)
	- [MNIST](./Library/keras/MNIST.md)

  - [PyTorch](./Library/torch.md)
	- [ カスタムデータセット化（画像）](./Library/torch/custum_data/custum_data.md)
  	- [ディープラーニング実装手順の基本（分類）](./Library/torch/DNN_torch.md)
	- [ディープラーニング実装手順の基本 カスタム数値データのデータセット化（回帰）](./Library/torch/torch_Linear_Regression.md)
	- [回帰分析 多項式回帰+単回帰分析（回帰）](./Library/torch/torch_Linears_Regression/torch_Linears_Regression.md)

	- [MNISTでDNNモデルの保存と読み込み](./Library/torch/MNIST_DNN.md)
	- [MNISTでDNNモデルの保存と読み込み 2（分類）](./Library/torch/DNN_MNIST/DNN_MNIST.md)
	- [NN LightGBM Titanic（分類）](./Library/torch/titanic/titanic.md)
	- [MNISTでCNN](./Library/torch/MNIST_CNN.md)
	- [MNISTでCNN 2（分類）](./Library/torch/CNN_MNIST_GPUdevice_Sequential/CNN_MNIST_GPUdevice_Sequential.md)
	- [GPU_CIFAR-10でCNN（分類）](./Library/torch/CIFAR_10_CNN_pytorch.md)
	- [転移学習 ResNet（分類）](./Library/torch/transferlearning/transferlearning.md)
	- [転移学習 VGG16 画像ファイル処理 os glob（分類）](./Library/torch/Transfer_Learning/TransferLearning.md)
	- [オートエンコーダ](./Library/torch/AutoEncoder/AutoEncoder.md)
	- [ResNet（転移学習の方がいいかな）](./Library/torch/ResNet.md)
	- [LSTM](./Library/torch/LSTM/LSTM.md)

	- [Tensors](./Library/torch/Tensors.md)
	- [Tensorsの操作と加算](./Library/torch/Tensorsmove.md)
		- [Tensorsのの新規作成](./Library/torch/Tensorsnews.md)
		- [Tensorsのサイズ取得とサイズ変更](./Library/torch/Tensorssize.md)
		- [テンソルの演算／計算](./Library/torch/Tensorsadd.md)
		- [テンソルの自動微分](./torch/Tensorsautoenco.md)
		- [テンソルのインデクシング／スライシング](./Library/torch/Tensorssli.md)
		- [テンソルからPython数値への変換](./Library/torch/Tensorspython.md)
		- [PyTorchテンソル ←→ NumPy多次元配列値、の変換＆連携](./Library/torch/Tensorsnp.md)
		- [PyTorchテンソル軸（データ格納）の順番を変更](./Library/torch/permute.md)
		- [テンソル演算でのGPU利用](./Library/torch/Tensorsgpu.md)
		- [Tensorsのサイズ取得とサイズ変更](./Library/torch/Tensorsmove.md)
		- [Tensorsの操作と演算のリンク T転置、インデックス付け、スライス、数学演算、線形代数、乱数などを含む100以上のテンソル演算](https://pytorch.org/docs/stable/torch.html)

	- [データローダーの作成(ミニバッチ学習の使用)](./Library/torch/DataLoader.md)
	- [ドロップアウトの適用](./torch/dropout.md)
	- [オプティマイザー（最適化用オブジェクト）の作成](./Library/torch/optimizer.md)
	- [損失関数の定義](./Library/torch/LossFunction.md)
	- [特定層の初期化関数](./Library/torch/initlayer.md)

	- [Autograd: 自動微分](./Library/torch/Autograd.md)
	- [動的な計算グラフの可視化 torchviz.makedot](./Library/torch/makedot.md)

	- [ニューロンのモデル設計と活性化関数・勾配計算](./Library/torch/NN_activation_gradiation.md)

  - [opencv](./Library/opencv.md)
	- **エッジ検出**
		- [Cannyエッジ検出](./Library/opencv/Canny.md)
	- [HOGとSVMによる物体検出](./Library/opencv/Object_detection_HoG_SVM.md)
	
  - scikit-image
	- 画像処理
		- [画像へのアクセス 行列](./Library/opencv/accimage.md)
		- [画像の読み込みとグレースケール](./Library/opencv/loadgrayimage.md)
		- [RGBチャネルの表示](./Library/opencv/RGB.md)
		- [RGBの2つの格納方法 planar formatとpacked format](./Library/opencv/planarpacked.md)
		- [RGBとBGR](./Library/opencv/RGBBGR.md)
		- [BGRとRGBの変換](./Library/opencv/RGBBGRchan.md)

		- [ヒストグラム計算](./Library/opencv/hist.md)
		- [統計量](./Library/opencv/dis.md)
		- [平均と分散の計算方法](./Library/opencv/ave.md)
		- [ガンマ変換](./Library/opencv/gamma.md)
		- [チャンネル別のガンマ変換](./Library/opencv/chanelgamma.md)
		- [疑似カラー表示](./Library/opencv/dummyclor.md)
		- [画像の演算（平均と重み付き平均）](./Library/opencv/imageave.md)
		- [アルファブレンディング](./Library/opencv/alpha.md)
		- [二値化・大津のしきい値と適応的二値化](./Library/opencv/2value.md)
		- [二値化ラベリングとモルフォロジー処理](./Library/opencv/label.md)
		- [細線化](./Library/opencv/distance.md)

		- [グレースケールへの変換](./Library/opencv/glayscale.md)
		- [画像の読み込み](./Library/opencv/loadimage.md)
		- **エッジ検出**
			- [グレースケール](./Library/opencv/gray.md)
			- [ガウシアンフィルタ 平滑化 ノイズ除去](./Library/opencv/GaussianBlur.md)

			- [関心領域](./Library/opencv/ROI.md)
	- **幾何変換**
		- [画像変形](./Library/opencv/imagechange.md)
		- [スケーリング・回転・鏡映](./Library/opencv/imageskale.md)
		- [射影変換](./Library/opencv/Projectivetransformation.md)
		- [順変換と逆変換](./Library/opencv/Forwardandinversetransformation.md)
		- [補間手法](./Library/opencv/Interpolationmethod.md)
	- **フィルタリング**
		- [平均値フィルタ](./Library/opencv/Averagefiltering.md)
		- [ガウシアンフィルタ](./Library/opencv/Gaussianfilter.md)
		- [ガボールフィルタ](./Library/opencv/Gaborfilter.md)
		- **微分フィルタ**
			- [ソーベルフィルタ(Sobel)，プレウィットフィルタ(Prewitt), ロバーツフィルタ(Roberts)](./Library/opencv/sobelprewittroberts.md)
				- [ソーベルフィルタ(Sobel)，プレウィットフィルタ(Prewitt)](./Library/opencv/sobelprewitt.md)
			- [ラプラシアンフィルタ](./Library/opencv/Laplacian.md)
				- [Laplacian of Gaussian （LoG）とゼロ交差](./Library/opencv/LaplacianLoG.md)
				- [Laplacian-of-GaussianとDifference-of-Gaussian](./Library/opencv/LoGDoG.md)
			- [Cannyエッジ](./Library/opencv/Cannyeddge.md)
			- [アンシャープマスキング](./Library/opencv/anshapemask.md)
		- **非線形フィルタ**
			- [非線形フィルタ:メディアンフィルタ](./Library/opencv/MedianFilter.md)
			- [非線形フィルタ:バイラテラルフィルタ ノンローカルミーンフィルタ](./Library/opencv/Non-localMeansFilter.md)
			- [画像の二次元フーリエ変換](./Library/opencv/Two-dimensionalFouriertransform.md)
		- **ローパスフィルタ**
			- [円形ボックスフィルタ](./Library/opencv/Circularboxfilter.md)
			- [ガウス型ローパスフィルタ](./Library/opencv/Gaussianlow-passfilter.md)
		- [ハイパスフィルタ](./Library/opencv/Highpassfilter.md)
			- [ガウス型ハイパスフィルタ](./Library/opencv/Gaussianhigh-passfilter.md)
		- [バンドパスフィルタ](./Library/opencv/Bandpassfilter.md)
		- [動画像のリアルタイムFFT](./Library/opencv/FFT2D.md)
	- **画像の劣化過程とその復元**
		- [ウィーナフィルタによる画像の復元](./Library/opencv/Wienerfilter.md)
		- [HDR合成](./Library/opencv/Image_degradation_process_andits_restoration/HDR.md)
	- [領域分割](./Library/opencv/Image_degradation_process_andits_restoration/Areadivision.md)
		- [k-meansクラスタリング](./Library/opencv/Image_degradation_process_andits_restoration/k-means.md)
			- [ガボール特徴も利用したkmeansクラスタリング](./Library/opencv/Image_degradation_process_andits_restoration/gabolk-means.md)
	- [動的輪郭モデル](./Library/opencv/Image_degradation_process_andits_restoration/Activecontourmodel.md)
	- [テンプレートマッチング](./Library/opencv/Image_degradation_process_andits_restoration/Templatematching.md)
	- [ハフ変換](./Library/opencv/Image_degradation_process_andits_restoration/Houghtransform.md)
	- [特徴点検出 （DoG・Fast・Harris・GFTT・AKAZE・BRISK・ORB）](./Library/opencv/Image_degradation_process_andits_restoration/Featurepointdetection.md)
	- [パノラマ画像作成](./Library/opencv/Image_degradation_process_andits_restoration/Panoramaimagecreation.md)

	- 音声処理
		- [音声データのフーリエ変換](./Library/opencv/Fouriertransformofvoicedata.md)
	- [短時間フーリエ変換によるスペクトログラムの表示](./Library/opencv/Short-timeFouriertransform.md)
	- [FFTと通常のフィルタリングの計算量の比較](./Library/opencv/FFTcomparison.md)

  - [nltk](./Library/nltk.md)

[戻る](./Overallview.md)

---

# 書き方フォーマット

# タイトル

### 書式

```python

```

### 引数

### 例


### 説明
