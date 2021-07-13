# データ分割
### データの読み込み
columnの作成.
``` python
df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash', 'Alcalinity of ash', 'Magnesium', 'Total phenols', 'Flavanoids', 'Nonflavabnoid phenols', 'Proanthocyanins', 'Color intensity','Hue', 'OD280/OD315 of diluted wines', 'Proline']
print('Class labels', np.unique(df_wine['Class label']))
print(df_wine.head())
```

### 特徴量とクラスラベルを別々に抽出
``` python
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
```

### 訓練データとテストデータに分割（全体の３０％をテストデータにする）
``` python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
```
## 尺度を揃えよう！
### 正則化と標準化
## 正則化とは
不良設定問題を解いたり過学習を防いだりするために、情報を追加する手法である。モデルの複雑さに罰則を科すために導入され、なめらかでないことに罰則をかけたり、パラメータのノルムの大きさに罰則をかけたりする。
詳細[https://ai-trend.jp/basic-study/neural-network/regularization/]
### min-maxスケーリングのインスタンス生成
``` python
mms = MinMaxScaler()
```
### 訓練データをスケーリング
``` python
X_train_norm = mms.fit_transform(X_train)
```
### テストデータをスケーリング
``` python
X_test_norm = mms.transform(X_test)
```
### 平均0 標準偏差1に変換
``` python
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)
```
- 多くの外れ値がある場合は`RobustScaler`を使用
- スケーリングを第１四分位数と第3四分位数に従ったデータセットのスケーリングと中央値の削除をおこなう。極端な値や外れ値が目立たなくなる

# 過学習
モデルが訓練データセットの観測結果にパラメーターを適合させすぎていて、新しいデータにうまく順応しない  
訓練データに対してモデルが複雑すぎることである  
### 対策
- さらに多くの訓練データを集める
- 正則化を通して、複雑さにペナルティを課す
- パラメータの数が少ない、より単純なモデルを選択する
- データの次元の数を減らす
### 前回はL2正則化を行ったが、今回はL1正則化を行う
L1正則化は無関係な特徴量の個数が多い高次元のデータセットであり、特に無関係な次元の数が訓練データよりも多い時である  
簡単な違いは2乗するかしないか
### 逆正則化パラメータC = 1.0はデフォルト
``` python
lr = LogisticRegression(penalty='l1', C=1.0, solver='liblinear', multi_class='ovr')
```
# 訓練データに適合
スコア計算.
``` python
lr.fit(X_train_std, y_train)
print('Training  accuracy:', lr.score(X_train_std, y_train))
print('Test accuracy:', lr.score(X_test_std, y_test))
```
