# LDAの概要

#### os

Windows10



#### 言語

C++



#### 必要条件

- RAM >= 1.0Gb
- CPU >= 2GHz



#### パラメータ推定

ギブスサンプリング



#### ダウンロード

以下のリンクからダウンロードできます．

http://sourceforge.net/projects/gibbslda



#### 入力データのフォーマット

訓練データ，テストデータともに以下の形式：

​		[M]

​		[document.1]

​		[document.2]

​		...

​		[document.M]

[M]は，文書数．それぞれの行は，1つの文書．[document.i]は文字数Niのi番目の文書となる．

[document.i] = [word.i1] [word.i2] ... [word.iNi]

[word.ij]は全て文字で，文字と文字の間は空白がある．



#### パラメータ推定

`$ lda -est [-alpha <double>] [-beta <double>] [-ntopics <int>] [-niters <int>] [-savestep <int>] [-twords <int>] -dfile <string>`

パラメータの[]はオプション

- `-est`: LDAのモデル推定
- `-alpha <double>`: LDAのハイパーパラメータαの値．デフォルトのαの値は50/K（Kはトピック数）
- `-beta <double>`：LDAのハイパーパラメータβの値．デフォルトのβの値は0.1
- `-ntopic <int>`：トピック数．デフォルトの値は100.
- `-niters <int>`：ギブスサンプリングの学習回数．デフォルトの値は2000
- `-savestep <int>`：ギブスサンプリングの学習の際，モデルを保存するステップ数．
- `-twords <int>`：それぞれのトピックで抽出する上位単語の単語数．デフォルトの値は0.
- `-dfile <string>`：訓練データファイル名



##### 出力ファイル

- `<model_name>.others`：モデルのパラメータの値が明記されているファイル
- `<model_name>.phi`：単語とトピックの分布，行がトピック，列が単語
- `<model_name>.theta`：トピックと文書の分布，行が文書，列がトピック
- `<model_name>.tassign`：単語のIDと単語が割り当てられたトピックを明記しているファイル
- `<model_name>.twords`：トピックに割り当てられた単語を明記しているファイル



#### テストデータの推論

`$ lda -inf -dir <string> -model <string> [-niters <int>] [-twords <int>] -dfile <string>`

- `-inf`: 学習済みLDAを使用したテストデータの推論
- `-dir <string>`: 学習済みLDAがあるディレクトリ名
- `-model <string>`: 学習済みLDAのモデル名
- `-niters <int>`: ギブスサンプリングの推論回数．デフォルトの値は20
- `-twords <int>`: それぞれのトピックで抽出する上位単語の単語数．デフォルトの値は0.
- `-dfile <string>`: テストデータファイル名



##### 出力ファイル

- `<newdocs>.dat.others`：モデルのパラメータの値が明記されているファイル
- `<newdocs>.dat.phi`：単語とトピックの分布，行がトピック，列が単語
- `<newdocs>.dat.theta`：トピックと文書の分布，行が文書，列がトピック
- `<newdocs>.dat.tassign`：単語のIDと単語が割り当てられたトピックを明記しているファイル
- `<newdocs>.dat.twords`：トピックに割り当てられた単語を明記しているファイル

