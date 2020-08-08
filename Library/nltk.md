# 英語による自然言語処理ライブラリ

---
title: NLTKの使い方をいろいろ調べてみた
tags: 自然言語処理 nltk Python
author: m__k
slide: false
---
Pythonで英語による自然言語処理をする上で役に立つNLTK(Natural Language Toolkit)の使い方をいろいろ調べてみたので、メモ用にまとめておきます。誰かのご参考になれば幸いです。

# 公式ドキュメント
- http://www.nltk.org/

# 参考記事
ほとんど以下の写経です。
- [Python,NLTKで自然言語処理](http://haya14busa.com/python-nltk-natural-language-processing/)
- [nltk](http://www.eonet.ne.jp/~tsugiyama/morph/morph01.html)

# インストール
pipで一発で入ります。

```
pip install nltk
```

# 使い方
## 機能のダウンロード
NLTKで形態素をするとき、最初に機能をダウンロードする必要があるとのこと。Pythonインタプリタで以下のコマンドを実行する必要があります。
以下のように `nltk.download` で機能を指定し、ダウンロードすると、  `$HOME` に `nltk_data` というフォルダが作成され、それぞれの機能に関するファイルが格納されます。

### 分かち書き（word_tokenize）

```py
import nltk
nltk.download('punkt')
```

### 品詞の取得（pos_tag）

```py
import nltk
nltk.download('averaged_perceptron_tagger')
```

### 注意！！

ググると 上記コマンドで `punkt` などの機能を指定せずに`nltk.download()` と実行すると、機能を選択しながらDLできる、みたいな記述がありますが、私の環境（MacBookPro）では `nltk.download()` を実行すると、Macが再起動します。

### 機能一括ダウンロード

毎回新しい機能を使うたびに`download`してはめんどくさいので、以下のコマンドで機能を一括でダウンロードできます。

```py

import nltk
nltk.download('all')
```

一括で機能を取得すると、 `nltk_data` のサイズが3.3GBになりました。

## 実行方法
### 分かち書き

```py

import nltk
s = "Hi, I'm Taro Yamada I woke up at 8am"
morph = nltk.word_tokenize(s)
print(morph)
# ['Hi', ',', 'I', "'m", 'Taro', 'Yamada', 'I', 'woke', 'up', 'at', '8am']
```

### 品詞の取得

```py
# 上のmorphに対して
pos = nltk.pos_tag(morph)
print(pos)
# [('Hi', 'NNP'), (',', ','), ('I', 'PRP'), ("'m", 'VBP'), ('Taro', 'JJ'), ('Yamada', 'NNP'), ('I', 'PRP'), ('woke', 'VBD'), ('up', 'RB'), ('at', 'IN'), ('8am', 'CD')]
```
TaroがJJ(形容詞)になってしまった...

#### 品詞タグの見方

|品詞タグ|品詞名（英語）|品詞名（日本語）|
|:-:|:-:|:-:|
|CC|Coordinating conjunction|調整接続詞|
| CD |Cardinal number|基数|
| DT|Determiner|限定詞|
| EX|Existential there|存在を表す there|
| FW|Foreign word|外国語|
| IN|Preposition or subordinating conjunction|前置詞または従属接続詞|
| JJ| Adjective| 形容詞|
| JJR|Adjective, comparative|	形容詞 (比較級)|
| JJS|Adjective, superlative|形容詞 (最上級)|
| LS|List item marker|-|
| MD| Modal|法|
| NN|Noun, singular or mass|名詞|
| NNS|Noun, plural|名詞 (複数形)|
| NNP|Proper noun, singular|固有名詞|
| NNPS|Proper noun, plural|固有名詞 (複数形)|
| PDT| Predeterminer|前限定辞|
| POS|Possessive ending|所有格の終わり|
| PRP|Personal pronoun| 人称代名詞 (PP)|
| PRP$|Possessive pronoun|所有代名詞 (PP$)|
| RB| Adverb| 副詞|
| RBR|Adverb, comparative|副詞 (比較級)|
| RBS| Adverb, superlative|副詞 (最上級)|
| RP| Particle |不変化詞|
| SYM| Symbol| 記号|
| TO|to|前置詞 to|
| UH|Interjection|感嘆詞|
|VB|Verb, base form|動詞 (原形)|
|VBD|Verb, past tense|動詞 (過去形)|
|VBG|Verb, gerund or present participle|動詞 (動名詞または現在分詞)|
|VBN|Verb, past participle|動詞 (過去分詞)|
|VBP|Verb, non-3rd person singular present|動詞 (三人称単数以外の現在形)|
|VBZ|Verb, 3rd person singular present|動詞 (三人称単数の現在形)|
|WDT|Wh-determiner|Wh 限定詞|
|WP|Wh-pronoun|Wh 代名詞|
|WP$|Possessive wh-pronoun|所有 Wh 代名詞|
|WRB|Wh-adverb|Wh 副詞|

## その他の機能
### 固有表現抽出(Named Entities)
Taro Yamada が人名、TokyoがGPE(? おそらく地名的なエンティティ)が取得できている

```py
import nltk
s = "Hi, I'm Taro Yamada I woke up at 8am"
morph = nltk.word_tokenize(s)
pos = nltk.pos_tag(morph)
# chunk.ne_chunkに品詞情報を渡す
entities = nltk.chunk.ne_chunk(pos)
print(entities)
#(S
# (GPE Hi/NNP)
#  ,/,
#  I/PRP
#  'm/VBP
#  (PERSON Taro/JJ Yamada/NNP)
#  ./.
#  I/PRP
#  woke/VBD
#  up/RB
#  at/IN
#  8am/CD)

# もうひとつ例
s = "I live in Tokyo."
morph = nltk.word_tokenize(s)
pos = nltk.pos_tag(morph)
entities = nltk.chunk.ne_chunk(pos)
print(entities)
# (S I/PRP live/VBP in/IN (GPE Tokyo/NNP) ./.)
```


### 見出し語(Lemmatisation)

```py

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
# 形態素のみを与える（文章（複数の形態素はダメ））
s = "getting"
print(lemmatizer.lemmatize(s))
print(lemmatizer.lemmatize(s, pos="n"))
print(lemmatizer.lemmatize(s, pos="v"))
# getting
# getting
# get

# 大文字が入ってるとだめ
s = "Getting"
print(lemmatizer.lemmatize(s))
print(lemmatizer.lemmatize(s, pos="n"))
print(lemmatizer.lemmatize(s, pos="v"))
# Getting
# Getting
# Getting
```

`pos` 引数に与えられる文字は以下の４つ

||引数に指定できる文字|
|:-:|:-:|
|NOUN|n|
|VERB|v|
|ADJ|a|
|ADV|r|

### 語幹化(Stemming)
方法が２つある（`PorterStemmer()`と`LancasterStemmer()`）

- PorterStemmer・・・一般的、安定的
- LancasterStemmer・・・アグレッシブ(?)


```py
from nltk import stem
s1 = "Getting"
s2 = "dialogue"
# PorterStemmer
stemmer = stem.PorterStemmer()
print(stemmer.stem(s1))
print(stemmer.stem(s2))
# get
# dialogu

# LancasterStemmer
stemmer2 = stem.LancasterStemmer()
print(stemmer2.stem(s1))
print(stemmer2.stem(s2))
# get
# dialog
```

### Sentence Tokenize

これでword tokenizeもできる

```python

from nltk import tokenize
s = '''He grinned and said, "I make lots of money.  On weekdays I receive
an average of 50 orders a day from all over the globe via the Internet."'
'''

# sentence
print(tokenize.sent_tokenize(s))
# ['He grinned and said, "I make lots of money.',
# 'On weekdays I receive\nan average of 50 orders a day from all over the globe via the Internet."\'']

# word
print(tokenize.word_tokenize(s))
print(tokenize.wordpunct_tokenize(s))
# ['He', 'grinned', 'and', 'said', ',', '``', 'I', 'make', 'lots', 'of', 'money', '.', 'On', 'weekdays', 'I', 'receive', 'an', 'average', 'of', '50', 'orders', 'a', 'day', 'from', 'all', 'over', 'the', 'globe', 'via', 'the', 'Internet', '.', "''", "'"]
# ['He', 'grinned', 'and', 'said', ',', '"', 'I', 'make', 'lots', 'of', 'money', '.', 'On', 'weekdays', 'I', 'receive', 'an', 'average', 'of', '50', 'orders', 'a', 'day', 'from', 'all', 'over', 'the', 'globe', 'via', 'the', 'Internet', '."\'']
```

### Stopwords
予め英語用のStopwordsが用意されている。
実際に使うときはこのStopwords集合を分かち書きに対してループで除去する処理を記述する必要あり

```py
from nltk.corpus import stopwords
stopset = set(stopwords.words('english'))
print(stopset)
#{'a',
# 'about',
# 'above',
# 'after',
# 'again',
# 'against',
#〜中略〜
# 'y',
# 'you',
# "you'd",
# "you'll",
# "you're",
# "you've",
# 'your',
# 'yours',
# 'yourself',
# 'yourselves'}
```

### 複合語の設定
使い方によってはユーザ辞書としても機能しそうだけど、その場合語幹化などは難しそう...

```Python
from nltk.tokenize import MWETokenizer

s = "I live in the United States . I have a lot of friends !"

# MWEインスタンス生成時に、以下のように複合語にしたいワードをタプル的に宣言（複数指定できる）
tokenizer = MWETokenizer([('the', 'United', 'States')])
# 追加で複合語を設定したければadd_mweでOK
tokenizer.add_mwe(('a', 'lot', 'of'))
print(tokenizer.tokenize(s.split()))
# ['I', 'live', 'in', 'the_United_States', '.', 'I', 'have', 'a_lot_of', 'friends', '!']

# このように複合語にピリオドなどの記号がくっついていると、正しく複合語と認識してくれない
ss = "I live in the United States. I have a lot of friends !"
print(tokenizer.tokenize(s.split()))
# ['I', 'live', 'in', 'the', 'United', 'States.', 'I', 'have', 'a_lot_of', 'friends', '!']
```

# 最後に
- NLTKだけでは例えば don't を do not に変換するとかはできないっぽい
- だいたいNLTKの機能がわかった気がする

おわり
