# PyTorch

> PyTorchは、次の2つの目的を持つPythonベースの科学計算パッケージです。 
- GPUのパワーを使用するためのNumPyの代替
- 最大の柔軟性とスピードを提供するディープラーニング研究プラットフォーム

使用するデータはテンソル型というNumpyと親和性の高いデータ型だけが使用できます。
numpy変換してテンソル型に変換するかデータセットを作成してテンソル型に変換するなどして学習を行います。

torchvisionは画像／ビデオ処理のPyTorch用追加パッケージで同時にインストールしないとパッケージ関係が不整合となるため、インストールしておく必要があります。（公式サイトでインストールコードを選択すれば一緒にインストールするためほぼ関係ないですが）

---
	versionは1.5.0+cu101

- [ カスタムデータセット化（画像）](./torch/custum_data/custum_data.md)
- [ディープラーニング実装手順の基本（分類）](./torch/DNN_torch.md)
- [ディープラーニング実装手順の基本 カスタム数値データのデータセット化（回帰）](./torch/torch_Linear_Regression.md)
- [回帰分析 多項式回帰+単回帰分析（回帰）](./torch/torch_Linears_Regression/torch_Linears_Regression.md)
- [MNISTでDNNモデルの保存と読み込み](./torch/MNIST_DNN.md)
- [MNISTでDNNモデルの保存と読み込み 2（分類）](./torch/DNN_MNIST/DNN_MNIST.md)
- [NN LightGBM Titanic（分類）](./torch/titanic/titanic.md)
- [MNISTでCNN](./torch/MNIST_CNN.md)
- [MNISTでCNN 2（分類）](./torch/CNN_MNIST_GPUdevice_Sequential/CNN_MNIST_GPUdevice_Sequential.md)
- [GPU_CIFAR-10でCNN（分類）](./torch/CIFAR_10_CNN_pytorch.md)
- [転移学習ResNet（分類）](./torch/transferlearning/transferlearning.md)
- [転移学習VGG16 画像ファイル処理 glob os（分類）](./torch/Transfer_Learning/TransferLearning.md)
- [オートエンコーダ](./torch/AutoEncoder/AutoEncoder.md)
- [ResNet](./torch/ResNet.md)
- [LSTM](./torch/LSTM/LSTM.md)
- [Tensors](./torch/Tensors.md)
	- [Tensorsのデータ型](./torch/Tensorsdatatype.md)
	- [Tensorsの操作と加算](./torch/Tensorsmove.md)
		- [Tensorsのの新規作成](./torch/Tensorsnews.md)
		- [Tensorsのサイズ取得とサイズ変更](./torch/Tensorssize.md)
		- [テンソルの演算／計算](./torch/Tensorsadd.md)
		- [テンソルの自動微分](./torch/Tensorsautoenco.md)
		- [テンソルのインデクシング／スライシング](./torch/Tensorssli.md)
		- [テンソルからPython数値への変換](./torch/Tensorspython.md)
		- [PyTorchテンソル ←→ NumPy多次元配列値、の変換＆連携](./torch/Tensorsnp.md)
		- [PyTorchテンソル軸（データ格納）の順番を変更](./torch/permute.md)
		- [テンソル演算でのGPU利用](./torch/Tensorsgpu.md)
	- [Tensorsの操作と演算のリンク T転置、インデックス付け、スライス、数学演算、線形代数、乱数などを含む100以上のテンソル演算](https://pytorch.org/docs/stable/torch.html)

- [データローダーの作成(ミニバッチ学習の使用)](./torch/DataLoader.md)
- [ドロップアウトの適用](./torch/dropout.md)
- [オプティマイザー（最適化用オブジェクト）の作成](./torch/optimizer.md)
- [損失関数の定義](./torch/LossFunction.md)
- [特定層の初期化関数](./torch/initlayer.md)

- [Autograd: 自動微分](./torch/Autograd.md)
- [動的な計算グラフの可視化 torchviz.makedot](./torch/makedot.md)

- [ニューロンのモデル設計と活性化関数と勾配計算](./torch/NN_activation_gradiation.md)

---

# 書き方フォーマット

# タイトル

### 書式

```python

```

### 引数

### 例


### 説明
