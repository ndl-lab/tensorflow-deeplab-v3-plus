# 図表抽出(Image extraction)


このプログラムは以下のリポジトリ(MITライセンス)を改変して作成しています。

[rishizek's repo](https://github.com/rishizek/tensorflow-deeplab-v3-plus).

## Setup
TensorFlow (r1.6)以降と Python 3をお使いください。

学習を試す場合は
tensorflowの[slim](https://github.com/tensorflow/models/tree/master/research/slim)から[resnet_v2_50_2017_04_14.tar.gz](http://download.tensorflow.org/models/resnet_v2_50_2017_04_14.tar.gz)をダウンロードし、
ini_checkpoints/resnet_v2_50
に配置してください。

推論のみ試す場合は、model50ディレクトリ下に
[学習済重みファイル](http://lab.ndl.go.jp/dataset/trainedweights.zip)を配置してください。


## Inference
```bash
python3 picture_extraction.py --input_dir INPUT_DIR --output_dir OUTPUT_DIR
```


## Training
pascal VOCのxmlのフォーマットで、図表部分の矩形領域に"4_illustration",資料全体の領域に"1_overall"のアノテーションを付与してください。
作成したxmlをpreprocess/annotxmlに、画像をpreprocess/imgに入れ、
preprocess/makeannotimage.py
を実行すると、セグメンテーション画像ファイルがpreprocess/annotimgに生成されます。

annotimg内にセグメンテーション画像が生成されたら、

```bash
python3 create_pascal_tf_record.py
python3 train_3class_101.py
```
を実行すると学習が始まります。



