# lightroom-selector
撮り溜めた写真をMLモデルに自動で選別させて、Lightroomの星とフラグを自動でつけるプラグイン


## 使い方

selector_pipeline.py
====================
Lightroom スマートプレビュー（JPEG, ~2048px）を対象に
1. 類似構図クラスタリング
2. 主題・構図・露出スコアリング
3. クラスタ内で上位カットを選定
4. XMP サイドカーに ★レーティングを書き込み
までを一気通貫で実行するスケルトン。

まだ TODO が多いけど、骨組みはこれでOK！
依存ライブラリ：opencv‑python, pillow, imagehash, numpy, scikit‑learn,
                  torch, ultralytics(yolov8), piexif (XMP 書込み用)

使い方：
$ python selector_pipeline.py --preview-dir ./SmartPreviews --output-dir ./xmp_out \
      --clusters 50 --topk 3

‑‑preview-dir  : Lightroom で生成したプレビューファイル群（JPEG）のルート
‑‑output-dir   : レーティングを書き込んだ XMP を吐き出す場所
‑‑clusters     : KMeans のクラスタ数（お好みで）
‑‑topk         : 各クラスタからピックする枚数
‑‑weights      : スコア重み JSON 文字列 (optional)