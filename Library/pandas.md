# Pandas

> データの集合を扱う<br>
> Pandasは<b>数値jデータの他に文字列データも簡易的に使用することができる</b>
> 表形式のデータ処理が得意です。


- [データフレームのメモリ削減関数](./Pandas/reduce_mem_usage.md)
- [複数データの結合](./Pandas/HomeCreditFeatureEngineering.md)

- <b>pandas-profiling(使用する場合は、pip installまたはconda installが必要 )</b>
    -  [各特徴量を個別に把握する](./Pandas/profilereport.md)
- <b>データ構造</b>
	- [Series](./Pandas/Series.md)
    	- [Seriesの作成](./Pandas/Seriesの作成.md)
    	- [データ・インデックスの抽出（Series）](./Pandas/data_indexget.md)
    	- [要素の追加（Series）](./Pandas/data_attend.md)
    	- [要素の削除（Series）](./Pandas/data_del.md)

	- [DataFrame](./Pandas/DataFrame.md)
    	- [DataFrameの作成](./Pandas/DataFrameの作成.md)
	    - [csvの読み込み](./Pandas/read_csv.md)
		- [csvの書き出し](./Pandas/write_csv.md)
		- [インデックス・カラムの取得・設定 index columns](./Pandas/data_indexset.md)
		- [カラムとローの省略拒否](./Library/Pandas/set_option.md)
		- [表結合 concat() merge()](./Pandas/concatmerge.md)

			- [縦連結](./Pandas/concat.md)
			- [結合 merge](./Pandas/merge.md)

		- [データの表示確認（リスト）](./Pandas/hyouzi.md)
		- [欠損値の有無の確認](./Pandas/isnull.md)
			- [**関数** 欠損値の有無の確認](./Pandas/missing_value_table.md)
		- [欠損値の削除 dropna](./Pandas/dropna.md)
		- [欠損値の補完 fillna](./Pandas/fillna.md)
		- [行か列の削除 drop](./Pandas/drop.md)
		- [特定のキーワードの変換 replace](./Pandas/replace.md)
		- [型の確認](./Pandas/type.md)
		- [特定の値をNA（欠損値）に変換 replace](./Pandas/replace.md)
		- [まとめて型変換 replace](./Pandas/all_replace.md)
		- [外れ値の変換 clip](./Pandas/clip.md)
		- [行と列数の確認](./Pandas/shape.md)
		- [列の名前の確認](./Pandas/colum.md)
		- [概要の確認](./Pandas/info.md)
		- [要約統計量の表示](./Pandas/describe.md)
		- [抽出・集計（特定の単語）](./Pandas/gotyuu.md)
		
		- [特定の区切りでデータを分割 binning](./Pandas/binning.md)
		- [要素の個数、頻度（出現回数）をカウント（Value_Counts()）](./Pandas/valuecounts.md)
		- [カテゴリー変数を質的変数に指定した値で変換（map()）](./Pandas/Library/Pandas/map.md)
		- [データから重複しないように人名を抽出](./Pandas/Library/Pandas/データから重複しないように人名を抽出.md)
		- [特定のカラムの型の出現個数の表示](./Pandas/Library/Pandas/objectnunique.md)
		- [Label encoding **and** One-hot encoding](./Pandas/Library/Pandas/LabelEncoder.md)
			- [One-Hot表現 ダミー変数に変換](./Pandas/Library/Pandas/one-hot.md)
		- [同じ値を持つデータをまとめて、それぞれの塊に対して共通の操作を行うgroupby](./Pandas/groupby.md)

		- <b>自然言語処理</b>
			- [小文字を大文字に、大文字を小文字に変換(upper lower)](./Pandas/upperlower.md)
			- [文字列の分割(split())](./Pandas/split.md)
			- [積集合$A\bigcap B$(a.intersection(b))](./Pandas/intersection.md)
			- [特定の文字列を分割して新特徴量にし、One-hot encoding](./Pandas/PredictFutureSales.md)
		
		- [行の連結 append()](./Pandas/行の連結.md)
		- [DataFrameの連結 concat()](./Pandas/連結.md)
		- [列の追加](./Pandas/列の追加.md)
		- [結合（マージ）](./Pandas/結合.md)
		- [同名でない列の結合（マージ）](./Pandas/同名でない結合.md)
		- [インデックスをKeyにして結合](./Pandas/インデックス結合.md)
		- [名前による参照](./Pandas/名前による参照.md)
		- [行または列の削除 .iloc dropna](./Pandas/行または列の削除.md)
		- [一部の行の取得（.head(), .tail()）](./Pandas/一部の行取得.md)
		- [要約統計量の取得 describe](./Pandas/要約統計量.md)
		- [行間列間の差を取得 diff](./Pandas/行間列間の差.md)
		- [グループ化](./Pandas/グループ化.md)

		- [フィルタリング](./Pandas/filterling.md)
	    - [indexを更新 再度indexを割り振る reset_index() set_index('カラム名')](./Pandas/index.md)
  
		- [参照](./Pandas/参照.md)
		- [ソート](./Pandas/ソート.md)

[全体に戻る](../Overallview.md)
