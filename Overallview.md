# $Python$
---
#### 項目は短く、$1$分で概要が把握できること
---
#### manコマンドの説明のような記述を心がけること
---
#### 例題を挿入し、用例を入れること
---
**$Tips$**
- [zipファイルのダウンロードと解凍、フォルダ作成](./load_zip.md)
- [多クラス分類表示](./Library/classifier_plot.md)
- [前処理と特徴量選択](./Library/feature_selection.md)
- [**時系列解析**](./時系列解析.md)
- [**コーパスのクリーニング** テキストを小文字に、角括弧内のテキスト、リンク、句読点数字を含む単語を削除、<br>単語の分割](./コーパスのクリーニング.md)
- [ストップワード除去](./stopwords.md)
- [**探索的データ解析EDA**](./EDA.md)


**$Memo$**
- [**組み込み関数**](./Embedded.md)
    - データタイプ
        - [数値型と基本操作](./Embedded/DataType.md)

        - [文字型と基本操作（split join）](./Embedded/Strings.md)
        - [リスト](./Embedded/list.md)
        - [辞書型](./Embedded/Dictionary.md)
        - [Boolean型](./Embedded/Boolean.md)
        - [タプル](./Embedded/Taples.md)
        - [Set](./Embedded/Set.md)
    - [型変換　キャスティング](./Embedded/Casting.md)
    - [演算子](./Embedded/Operator.md)
    - [if文](./Embedded/if.md)
    - [Loop文](./Embedded/for.md)
        - [イテラブル(Iterable) と イテレータ(Iterator)](./Embedded/iterable.md)
    - [関数 (Function)](./Embedded/def.md)

- [**ライブラリ**](./Library.md)
    - [numpy](./Library/numpy.md)
    - [pandas](./Library/pandas.md)
    - [Matplotlib（日本語使用：pip install japanize-matplotlib）import japanize_matplotlib](./Library/Matplotlib.md)
        - [seaboen](./Library/seaborn.md)
    - [scikit-learn](./Library/scikit-learn.md)
    - [statsmodels](./Library/statsmodels.md)

    - [TensorFlow](./Library/tensorflow.md)
    - [keras](./Library/keras.md)
    - [PyTorch](./Library/torch.md)

    - [opencv](./Library/opencv.md)

    - [XGBClassifier](./Library/scikit-learn/xgboost_XGBClassifier.md)

    - [nltk](./Library/nltk.md)



---
# 書き方フォーマット

# タイトル

名前

    less - more の反対 (more のようなページャー)

### 書式

```python

```

### 引数

    以降の説明で ^X は control-X を意味する。 ESC は ESCAPE キー
    である。例えば、ESC-v は "ESCAPE" を押した後に "v" を 押す
    という意味である。

    h または H
    　　ヘルプ。コマンドの概要を表示する。もし、他のコマンド
    　　を忘れた時は、このコマンドを思い出すこと。

    SPACE または ^V または f または ^F
    　　前方に 1 ウインドウ分 (N ・・

### 例

    man less

### 説明

    less は more (1) と同様なプログラムであるが、ファイル内 で
    の・・

---