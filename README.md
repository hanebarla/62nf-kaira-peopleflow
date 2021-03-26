# 62nf-kaira-peopleflow

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hanebarla/62nf-kaira-peopleflow/blob/main/demo.ipynb)

## デモの始め方
1. この[リンク](https://colab.research.google.com/github/hanebarla/62nf-kaira-peopleflow/blob/main/demo.ipynb)からGoogle Colabを開きます。
1. GPUを使えるようにします。`ランタイム > ランタイムのタイプを変更 > ハードウェアアクセラレータ > GPU `と設定してください。
1. 次にスクリプトを実行します。`ランタイム > すべてのセルを実行(Ctr + F9) `を実行してください。

### 注意点
* USBカメラでの動作のみを確認しています。

### 出力
![カメラ画像](imgs/4896_004740.jpg)
一番上の出力はカメラで取り込んでいる画像を示しています。

![密度画像](imgs/pred_dense.png)
人がどこに集まっているのかを示しています。
人が集まているほど赤に近づきます。

![人流画像](imgs/pred_quiver.png)
8方向で人の流れを示しています。
色の濃いほどより多くの人が流れています。

***
## 解説

***
## ありそうなQ&A
