# Seaborn

> データのきれいに視覚化するための機能が豊富なライブラリmatplotlibと併用することが多そう
> ```python
> import matplotlib as mpl
> import matplotlib.pyplot as plt
> import seaborn as sns
> import japanize_matplotlib
> 
> sns.countplot(x='Survived', data=train)
> plt.title('死亡者と生存者の数')
> plt.xticks([0,1],['死亡者', '生存者'])
> plt.show()
> ```
> を使うだけでグラフを簡単に描画できる
> 作成したグラフは、matplotlib.pyplot.show()で表示できる。

- <b>グラフ表示</b>
    - [集計表とクロス集計](./Seaborn/集計表.md)
    - [ヒストグラム sns.distplot()](./Seaborn/hist.md)
    - [散布図+ヒストグラム ペアプロット sns.jointplot() sns.pairplot()](./Seaborn/joinplotpaorplot/joinplotpaorplot.md)
    - [カテゴリカルプロット sns.barplot() sns.countplot() sns.boxplot() sns.violinplot() sns.swarmplot](./Seaborn/categorical/categorical.md)
    - [ヒートマップ sns.heatmap()](./Seaborn/heatmap/heatmap.md)
    - [スタイル変更 pltとの連携 sns.set()](./Seaborn/set/set.md)

[戻る](../Overallview.md)