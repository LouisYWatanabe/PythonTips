# '-'の前後で文字列を抽出して新しい特徴量を作る

```python
print(item_categories.shape)
item_categories.head()
```

    (84, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_category_name</th>
      <th>item_category_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PC - Гарнитуры/Наушники</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Аксессуары - PS2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Аксессуары - PS3</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Аксессуары - PS4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Аксессуары - PSP</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
item_categorie = item_categories['item_category_name'].unique()
item_categorie
```




    array(['PC - Гарнитуры/Наушники', 'Аксессуары - PS2', 'Аксессуары - PS3',
           'Аксессуары - PS4', 'Аксессуары - PSP', 'Аксессуары - PSVita',
           'Аксессуары - XBOX 360', 'Аксессуары - XBOX ONE', 'Билеты (Цифра)',
           'Доставка товара', 'Игровые консоли - PS2',
           'Игровые консоли - PS3', 'Игровые консоли - PS4',
           'Игровые консоли - PSP', 'Игровые консоли - PSVita',
           'Игровые консоли - XBOX 360', 'Игровые консоли - XBOX ONE',
           'Игровые консоли - Прочие', 'Игры - PS2', 'Игры - PS3',
           'Игры - PS4', 'Игры - PSP', 'Игры - PSVita', 'Игры - XBOX 360',
           'Игры - XBOX ONE', 'Игры - Аксессуары для игр',
           'Игры Android - Цифра', 'Игры MAC - Цифра',
           'Игры PC - Дополнительные издания',
           'Игры PC - Коллекционные издания', 'Игры PC - Стандартные издания',
           'Игры PC - Цифра', 'Карты оплаты (Кино, Музыка, Игры)',
           'Карты оплаты - Live!', 'Карты оплаты - Live! (Цифра)',
           'Карты оплаты - PSN', 'Карты оплаты - Windows (Цифра)',
           'Кино - Blu-Ray', 'Кино - Blu-Ray 3D', 'Кино - Blu-Ray 4K',
           'Кино - DVD', 'Кино - Коллекционное',
           'Книги - Артбуки, энциклопедии', 'Книги - Аудиокниги',
           'Книги - Аудиокниги (Цифра)', 'Книги - Аудиокниги 1С',
           'Книги - Бизнес литература', 'Книги - Комиксы, манга',
           'Книги - Компьютерная литература',
           'Книги - Методические материалы 1С', 'Книги - Открытки',
           'Книги - Познавательная литература', 'Книги - Путеводители',
           'Книги - Художественная литература', 'Книги - Цифра',
           'Музыка - CD локального производства',
           'Музыка - CD фирменного производства', 'Музыка - MP3',
           'Музыка - Винил', 'Музыка - Музыкальное видео',
           'Музыка - Подарочные издания', 'Подарки - Атрибутика',
           'Подарки - Гаджеты, роботы, спорт', 'Подарки - Мягкие игрушки',
           'Подарки - Настольные игры',
           'Подарки - Настольные игры (компактные)',
           'Подарки - Открытки, наклейки', 'Подарки - Развитие',
           'Подарки - Сертификаты, услуги', 'Подарки - Сувениры',
           'Подарки - Сувениры (в навеску)',
           'Подарки - Сумки, Альбомы, Коврики д/мыши', 'Подарки - Фигурки',
           'Программы - 1С:Предприятие 8', 'Программы - MAC (Цифра)',
           'Программы - Для дома и офиса',
           'Программы - Для дома и офиса (Цифра)', 'Программы - Обучающие',
           'Программы - Обучающие (Цифра)', 'Служебные', 'Служебные - Билеты',
           'Чистые носители (шпиль)', 'Чистые носители (штучные)',
           'Элементы питания'], dtype=object)



`item_category_name`は「タイプ-サブタイプ」の構成になっています。<br>
`type`と`subtype`を新しい特徴量として追加します。


```python
# '-'でカテゴリ名を分割
item_categories['split'] = item_categories['item_category_name'].str.split('-')
# typeには-で分割した先頭の値を代入
item_categories['type'] = item_categories['split'].map(lambda x:x[0].strip())
# sub_typeには-で分割した2番目の値を代入、sub-typeには、typeのデータをsub_typeとして代入
item_categories['sub_type'] = item_categories['split'].map(lambda x:x[1].strip() if len(x) > 1 else x[0].strip())
item_categories.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_category_name</th>
      <th>item_category_id</th>
      <th>split</th>
      <th>type</th>
      <th>sub_type</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PC - Гарнитуры/Наушники</td>
      <td>0</td>
      <td>[PC ,  Гарнитуры/Наушники]</td>
      <td>PC</td>
      <td>Гарнитуры/Наушники</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Аксессуары - PS2</td>
      <td>1</td>
      <td>[Аксессуары ,  PS2]</td>
      <td>Аксессуары</td>
      <td>PS2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Аксессуары - PS3</td>
      <td>2</td>
      <td>[Аксессуары ,  PS3]</td>
      <td>Аксессуары</td>
      <td>PS3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Аксессуары - PS4</td>
      <td>3</td>
      <td>[Аксессуары ,  PS4]</td>
      <td>Аксессуары</td>
      <td>PS4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Аксессуары - PSP</td>
      <td>4</td>
      <td>[Аксессуары ,  PSP]</td>
      <td>Аксессуары</td>
      <td>PSP</td>
    </tr>
  </tbody>
</table>
</div>




```python
item_categories['type'].value_counts()
```




    Книги                                13
    Подарки                              12
    Игровые консоли                       8
    Игры                                  8
    Аксессуары                            7
    Музыка                                6
    Программы                             6
    Кино                                  5
    Карты оплаты                          4
    Игры PC                               4
    Служебные                             2
    Доставка товара                       1
    PC                                    1
    Чистые носители (шпиль)               1
    Билеты (Цифра)                        1
    Элементы питания                      1
    Карты оплаты (Кино, Музыка, Игры)     1
    Игры MAC                              1
    Игры Android                          1
    Чистые носители (штучные)             1
    Name: type, dtype: int64




```python
item_categories['sub_type'].value_counts()
```




    Цифра                         4
    PS4                           3
    PSVita                        3
    XBOX ONE                      3
    Blu                           3
                                 ..
    1С:Предприятие 8              1
    Подарочные издания            1
    CD фирменного производства    1
    Компьютерная литература       1
    Фигурки                       1
    Name: sub_type, Length: 65, dtype: int64




```python
# typeをone-hot encodingする
types = pd.DataFrame(item_categories['type'])
# one-hot encoding
types = pd.get_dummies(types)

# shops, city_onehotを横方向に連結
item_categories = pd.concat([item_categories, types], axis=1)
# shopsからcity_nameカラムを削除
item_categories = item_categories.drop('type', axis=1)
item_categories.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_category_name</th>
      <th>item_category_id</th>
      <th>split</th>
      <th>sub_type</th>
      <th>type_PC</th>
      <th>type_Аксессуары</th>
      <th>type_Билеты (Цифра)</th>
      <th>type_Доставка товара</th>
      <th>type_Игровые консоли</th>
      <th>type_Игры</th>
      <th>...</th>
      <th>type_Карты оплаты (Кино, Музыка, Игры)</th>
      <th>type_Кино</th>
      <th>type_Книги</th>
      <th>type_Музыка</th>
      <th>type_Подарки</th>
      <th>type_Программы</th>
      <th>type_Служебные</th>
      <th>type_Чистые носители (шпиль)</th>
      <th>type_Чистые носители (штучные)</th>
      <th>type_Элементы питания</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PC - Гарнитуры/Наушники</td>
      <td>0</td>
      <td>[PC ,  Гарнитуры/Наушники]</td>
      <td>Гарнитуры/Наушники</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Аксессуары - PS2</td>
      <td>1</td>
      <td>[Аксессуары ,  PS2]</td>
      <td>PS2</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Аксессуары - PS3</td>
      <td>2</td>
      <td>[Аксессуары ,  PS3]</td>
      <td>PS3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Аксессуары - PS4</td>
      <td>3</td>
      <td>[Аксессуары ,  PS4]</td>
      <td>PS4</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Аксессуары - PSP</td>
      <td>4</td>
      <td>[Аксессуары ,  PSP]</td>
      <td>PSP</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 24 columns</p>
</div>




```python
# sub_typeをone-hot encodingする
sub_types = pd.DataFrame(item_categories['sub_type'])
# one-hot encoding
sub_types = pd.get_dummies(sub_types)

# shops, city_onehotを横方向に連結
item_categories = pd.concat([item_categories, sub_types], axis=1)
# shopsからcity_nameカラムを削除
item_categories = item_categories.drop('sub_type', axis=1)
item_categories.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>item_category_name</th>
      <th>item_category_id</th>
      <th>split</th>
      <th>type_PC</th>
      <th>type_Аксессуары</th>
      <th>type_Билеты (Цифра)</th>
      <th>type_Доставка товара</th>
      <th>type_Игровые консоли</th>
      <th>type_Игры</th>
      <th>type_Игры Android</th>
      <th>...</th>
      <th>sub_type_Стандартные издания</th>
      <th>sub_type_Сувениры</th>
      <th>sub_type_Сувениры (в навеску)</th>
      <th>sub_type_Сумки, Альбомы, Коврики д/мыши</th>
      <th>sub_type_Фигурки</th>
      <th>sub_type_Художественная литература</th>
      <th>sub_type_Цифра</th>
      <th>sub_type_Чистые носители (шпиль)</th>
      <th>sub_type_Чистые носители (штучные)</th>
      <th>sub_type_Элементы питания</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>PC - Гарнитуры/Наушники</td>
      <td>0</td>
      <td>[PC ,  Гарнитуры/Наушники]</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Аксессуары - PS2</td>
      <td>1</td>
      <td>[Аксессуары ,  PS2]</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Аксессуары - PS3</td>
      <td>2</td>
      <td>[Аксессуары ,  PS3]</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Аксессуары - PS4</td>
      <td>3</td>
      <td>[Аксессуары ,  PS4]</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Аксессуары - PSP</td>
      <td>4</td>
      <td>[Аксессуары ,  PSP]</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 88 columns</p>
</div>



# 先頭から一番初めの半角スペースまでの文字列を抽出する

```python
print(shops.shape)
shops
```

    (60, 2)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shop_name</th>
      <th>shop_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>!Якутск Орджоникидзе, 56 фран</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>!Якутск ТЦ "Центральный" фран</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Адыгея ТЦ "Мега"</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Балашиха ТРК "Октябрь-Киномир"</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Волжский ТЦ "Волга Молл"</td>
      <td>4</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Вологда ТРЦ "Мармелад"</td>
      <td>5</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Воронеж (Плехановская, 13)</td>
      <td>6</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Воронеж ТРЦ "Максимир"</td>
      <td>7</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Воронеж ТРЦ Сити-Парк "Град"</td>
      <td>8</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Выездная Торговля</td>
      <td>9</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Жуковский ул. Чкалова 39м?</td>
      <td>10</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Жуковский ул. Чкалова 39м²</td>
      <td>11</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Интернет-магазин ЧС</td>
      <td>12</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Казань ТЦ "Бехетле"</td>
      <td>13</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Казань ТЦ "ПаркХаус" II</td>
      <td>14</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Калуга ТРЦ "XXI век"</td>
      <td>15</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Коломна ТЦ "Рио"</td>
      <td>16</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Красноярск ТЦ "Взлетка Плаза"</td>
      <td>17</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Красноярск ТЦ "Июнь"</td>
      <td>18</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Курск ТЦ "Пушкинский"</td>
      <td>19</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Москва "Распродажа"</td>
      <td>20</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Москва МТРЦ "Афи Молл"</td>
      <td>21</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Москва Магазин С21</td>
      <td>22</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Москва ТК "Буденовский" (пав.А2)</td>
      <td>23</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Москва ТК "Буденовский" (пав.К7)</td>
      <td>24</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Москва ТРК "Атриум"</td>
      <td>25</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Москва ТЦ "Ареал" (Беляево)</td>
      <td>26</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Москва ТЦ "МЕГА Белая Дача II"</td>
      <td>27</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Москва ТЦ "МЕГА Теплый Стан" II</td>
      <td>28</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Москва ТЦ "Новый век" (Новокосино)</td>
      <td>29</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Москва ТЦ "Перловский"</td>
      <td>30</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Москва ТЦ "Семеновский"</td>
      <td>31</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Москва ТЦ "Серебряный Дом"</td>
      <td>32</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Мытищи ТРК "XL-3"</td>
      <td>33</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Н.Новгород ТРЦ "РИО"</td>
      <td>34</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Н.Новгород ТРЦ "Фантастика"</td>
      <td>35</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Новосибирск ТРЦ "Галерея Новосибирск"</td>
      <td>36</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Новосибирск ТЦ "Мега"</td>
      <td>37</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Омск ТЦ "Мега"</td>
      <td>38</td>
    </tr>
    <tr>
      <th>39</th>
      <td>РостовНаДону ТРК "Мегацентр Горизонт"</td>
      <td>39</td>
    </tr>
    <tr>
      <th>40</th>
      <td>РостовНаДону ТРК "Мегацентр Горизонт" Островной</td>
      <td>40</td>
    </tr>
    <tr>
      <th>41</th>
      <td>РостовНаДону ТЦ "Мега"</td>
      <td>41</td>
    </tr>
    <tr>
      <th>42</th>
      <td>СПб ТК "Невский Центр"</td>
      <td>42</td>
    </tr>
    <tr>
      <th>43</th>
      <td>СПб ТК "Сенная"</td>
      <td>43</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Самара ТЦ "Мелодия"</td>
      <td>44</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Самара ТЦ "ПаркХаус"</td>
      <td>45</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Сергиев Посад ТЦ "7Я"</td>
      <td>46</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Сургут ТРЦ "Сити Молл"</td>
      <td>47</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Томск ТРЦ "Изумрудный Город"</td>
      <td>48</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Тюмень ТРЦ "Кристалл"</td>
      <td>49</td>
    </tr>
    <tr>
      <th>50</th>
      <td>Тюмень ТЦ "Гудвин"</td>
      <td>50</td>
    </tr>
    <tr>
      <th>51</th>
      <td>Тюмень ТЦ "Зеленый Берег"</td>
      <td>51</td>
    </tr>
    <tr>
      <th>52</th>
      <td>Уфа ТК "Центральный"</td>
      <td>52</td>
    </tr>
    <tr>
      <th>53</th>
      <td>Уфа ТЦ "Семья" 2</td>
      <td>53</td>
    </tr>
    <tr>
      <th>54</th>
      <td>Химки ТЦ "Мега"</td>
      <td>54</td>
    </tr>
    <tr>
      <th>55</th>
      <td>Цифровой склад 1С-Онлайн</td>
      <td>55</td>
    </tr>
    <tr>
      <th>56</th>
      <td>Чехов ТРЦ "Карнавал"</td>
      <td>56</td>
    </tr>
    <tr>
      <th>57</th>
      <td>Якутск Орджоникидзе, 56</td>
      <td>57</td>
    </tr>
    <tr>
      <th>58</th>
      <td>Якутск ТЦ "Центральный"</td>
      <td>58</td>
    </tr>
    <tr>
      <th>59</th>
      <td>Ярославль ТЦ "Альтаир"</td>
      <td>59</td>
    </tr>
  </tbody>
</table>
</div>



shop_idで同じshop_nameをタイプミス？で登録されています。<br>重複しているshop_nameに対応するsho_idを統一します。


```python
# shop_idの統一
# マージ後を考え、`sales_train`と`test`に対してshop_id = 0 を shop_id = 57に shop_id = 1 を shop_id = 58 に shop_id = 10 を shop_id = 11に変換する
sales_train.loc[sales_train['shop_id'] == 0, 'shop_id'] = 57
test.loc[test['shop_id'] == 0, 'shop_id'] = 57

sales_train.loc[sales_train['shop_id'] == 1, 'shop_id'] = 58
test.loc[test['shop_id'] == 1, 'shop_id'] = 58

sales_train.loc[sales_train['shop_id'] == 10, 'shop_id'] = 11
test.loc[test['shop_id'] == 10, 'shop_id'] = 11
```

shop_nameはロシアの各都市名 半角スペース タイプ 半角スペース 店名のような構成です（Москваはモスクワ）。<br>最初のスペースまでを抽出し、city_nameとして追加します。（たぶん都市名が抜けているデータが何個かあるようだが今のところ無視）<br>この特徴量をOne-Hot encodingします。


```python
# shop_name先頭の!を削除
shops.loc[shops['shop_name'] == '!Якутск Орджоникидзе, 56 фран', 'shop_name'] = 'Якутск Орджоникидзе, 56 фран'
shops.loc[shops['shop_name'] == '!Якутск ТЦ "Центральный" фран', 'shop_name'] = 'кутск ТЦ "Центральный" фран'

# shop_nameの先頭を抽出してcity_nameを追加
shops['city_name'] = shops['shop_name'].str.split(' ').map(lambda x : x[0])    # 先頭から一番初めの半角スペースまでの文字列を抽出
shops.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shop_name</th>
      <th>shop_id</th>
      <th>city_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Якутск Орджоникидзе, 56 фран</td>
      <td>0</td>
      <td>Якутск</td>
    </tr>
    <tr>
      <th>1</th>
      <td>кутск ТЦ "Центральный" фран</td>
      <td>1</td>
      <td>кутск</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Адыгея ТЦ "Мега"</td>
      <td>2</td>
      <td>Адыгея</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Балашиха ТРК "Октябрь-Киномир"</td>
      <td>3</td>
      <td>Балашиха</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Волжский ТЦ "Волга Молл"</td>
      <td>4</td>
      <td>Волжский</td>
    </tr>
  </tbody>
</table>
</div>




```python
city_name = pd.DataFrame(shops['city_name'])
# one-hot encoding
city_onehot = pd.get_dummies(city_name)

# shops, city_onehotを横方向に連結
shops = pd.concat([shops, city_onehot], axis=1)
# shopsからcity_nameカラムを削除
shops = shops.drop('city_name', axis=1)
shops.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>shop_name</th>
      <th>shop_id</th>
      <th>city_name_Адыгея</th>
      <th>city_name_Балашиха</th>
      <th>city_name_Волжский</th>
      <th>city_name_Вологда</th>
      <th>city_name_Воронеж</th>
      <th>city_name_Выездная</th>
      <th>city_name_Жуковский</th>
      <th>city_name_Интернет-магазин</th>
      <th>...</th>
      <th>city_name_Сургут</th>
      <th>city_name_Томск</th>
      <th>city_name_Тюмень</th>
      <th>city_name_Уфа</th>
      <th>city_name_Химки</th>
      <th>city_name_Цифровой</th>
      <th>city_name_Чехов</th>
      <th>city_name_Якутск</th>
      <th>city_name_Ярославль</th>
      <th>city_name_кутск</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Якутск Орджоникидзе, 56 фран</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>кутск ТЦ "Центральный" фран</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Адыгея ТЦ "Мега"</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Балашиха ТРК "Октябрь-Киномир"</td>
      <td>3</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Волжский ТЦ "Волга Молл"</td>
      <td>4</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 66 columns</p>
</div>

