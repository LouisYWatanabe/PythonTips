# opencv<br>scikit-image

> 画像処理ライブラリ

  - [opencv](./opencv.md)
	- [画像の表示 グレースケール imread() cvtColor(image, cv2.COLOR_BGR2RGB)](./opencv/imread/imread.md)
	- [cropped 切り取り 画像の保存 image cv2.imwrite()](./opencv/slicing/slicing.md)
	- [OpenCVによる二値化 単純に閾値を指定して2値化 大津の2値化（Otsu's binarization） Adaptive Thresholding cv2.threshold() cv2.adaptiveThreshold()](./opencv/binarization/binarization.md)

  	- **エッジ検出**
		- [Cannyエッジ検出](./opencv/Canny.md)
	- [HOGとSVMによる物体検出](./opencv/Object_detection_HoG_SVM.md)
	
  - scikit-image
	- 画像処理
		- [画像へのアクセス 行列](./opencv/accimage.md)
		- [画像の読み込みとグレースケール](./opencv/loadgrayimage.md)
		- [RGBチャネルの表示](./opencv/RGB.md)
		- [RGBの2つの格納方法 planar formatとpacked format](./opencv/planarpacked.md)
		
		- [RGBとBGR](./opencv/RGBBGR.md)

		- [ヒストグラム計算](./opencv/hist.md)
		- [統計量](./opencv/dis.md)
		- [平均と分散の計算方法](./opencv/ave.md)
		- [ガンマ変換](./opencv/gamma.md)
		- [チャンネル別のガンマ変換](./opencv/chanelgamma.md)
		- [疑似カラー表示](./opencv/dummyclor.md)
		- [画像の演算（平均と重み付き平均）](./opencv/imageave.md)
		- [アルファブレンディング](./opencv/alpha.md)
		- [二値化・大津のしきい値と適応的二値化](./opencv/2value.md)
		- [二値化ラベリングとモルフォロジー処理](./opencv/label.md)
		- [細線化](./opencv/distance.md)
		- [グレースケールへの変換](./opencv/glayscale.md)
		- [画像の読み込み](./opencv/loadimage.md)
		- **エッジ検出**
			- [グレースケール](./opencv/gray.md)
			- [ガウシアンフィルタ 平滑化 ノイズ除去](./opencv/GaussianBlur.md)
			- [関心領域](./opencv/ROI.md)

	- **幾何変換**
		- [画像変形](./opencv/imagechange.md)
		- [スケーリング・回転・鏡映](./opencv/imageskale.md)
		- [射影変換](./opencv/Projectivetransformation.md)
		- [順変換と逆変換](./opencv/Forwardandinversetransformation.md)
		- [補間手法](./opencv/Interpolationmethod.md)
	- **フィルタリング**
		- [平均値フィルタ](./opencv/Averagefiltering.md)
		- [ガウシアンフィルタ](./opencv/Gaussianfilter.md)
		- [ガボールフィルタ](./opencv/Gaborfilter.md)
		- **微分フィルタ**
			- [ソーベルフィルタ(Sobel)，プレウィットフィルタ(Prewitt), ロバーツフィルタ(Roberts)](./opencv/sobelprewittroberts.md)
				- [ソーベルフィルタ(Sobel)，プレウィットフィルタ(Prewitt)](./opencv/sobelprewitt.md)
			- [ラプラシアンフィルタ](./opencv/Laplacian.md)
				- [Laplacian of Gaussian （LoG）とゼロ交差](./opencv/LaplacianLoG.md)
				- [Laplacian-of-GaussianとDifference-of-Gaussian](./opencv/LoGDoG.md)
			- [Cannyエッジ](./opencv/Cannyeddge.md)
			- [アンシャープマスキング](./opencv/anshapemask.md)
		- **非線形フィルタ**
			- [非線形フィルタ:メディアンフィルタ](./opencv/MedianFilter.md)
			- [非線形フィルタ:バイラテラルフィルタ ノンローカルミーンフィルター](./opencv/Non-localMeansFilter.md)
			- [画像の二次元フーリエ変換](./opencv/Two-dimensionalFouriertransform.md)
		- **ローパスフィルタ**
			- [円形ボックスフィルタ](./opencv/Circularboxfilter.md)
			- [ガウス型ローパスフィルタ](./opencv/Gaussianlow-passfilter.md)
		- [ハイパスフィルタ](./opencv/Highpassfilter.md)
			- [ガウス型ハイパスフィルタ](./opencv/Gaussianhigh-passfilter.md)
		- [バンドパスフィルタ](./opencv/Bandpassfilter.md)
		- [動画像のリアルタイムFFT](./opencv/FFT2D.md)
	- **画像の劣化過程とその復元**
		- [ウィーナフィルタによる画像の復元](./opencv/Wienerfilter.md)
		- [HDR合成](./opencv/Image_degradation_process_andits_restoration/HDR.md)
	- [領域分割](./opencv/Image_degradation_process_andits_restoration/Areadivision.md)
		- [k-meansクラスタリング](./opencv/Image_degradation_process_andits_restoration/k-means.md)
		- [ガボール特徴も利用したkmeansクラスタリング](./opencv/Image_degradation_process_andits_restoration/gabolk-means.md)
	- [動的輪郭モデル](./opencv/Image_degradation_process_andits_restoration/Activecontourmodel.md)
	- [テンプレートマッチング](./opencv/Image_degradation_process_andits_restoration/Templatematching.md)
	- [ハフ変換](./opencv/Image_degradation_process_andits_restoration/Houghtransform.md)
	- [特徴点検出 （DoG・Fast・Harris・GFTT・AKAZE・BRISK・ORB）](./opencv/Image_degradation_process_andits_restoration/Featurepointdetection.md)
	- [パノラマ画像作成](./opencv/Image_degradation_process_andits_restoration/Panoramaimagecreation.md)

	- 音声処理
		- [音声データのフーリエ変換](./opencv/Fouriertransformofvoicedata.md)
	- [短時間フーリエ変換によるスペクトログラムの表示](./opencv/Short-timeFouriertransform.md)
	- [FFTと通常のフィルタリングの計算量の比較](./opencv/FFTcomparison.md)

[全体に戻る](../Overallview.md)



#### インストール方法

- main (core) モジュールのみで良い場合:
```
$ pip install opencv-python
```
- contrib (extra) モジュールも必要な場合:
```
$ pip install opencv-contrib-python
```
main/extra モジュールの分類についてはこちら．OpenCV の使いたい機能に応じて選択する．

商用利用しないのであれば，main モジュールが包含される opencv-contrib-python の方にしておけば良いと思います．
```