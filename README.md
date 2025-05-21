# lightroom-selector
撮り溜めた写真をMLモデルに自動で選別させて、Lightroomの星とフラグを自動でつけるプラグイン

## 要件

要件 | 具体化
---|---
Lightroom Classic | Lightroom CC ではなく Classic 版を使用
Lightroom クラウド版 で溜めた RAW を対象 | Classic に同期 → スマートプレビュー生成 が最速＆公式 API 不要
I/O を抑えつつ 1,000+ 枚をふるいにかける | プレビュー JPEG (~2 MP) だけを Python で読む
スコア 4 層 (構図 > 主題ピント > 三分割距離 > ヒストグラム) | score = w1*t1 + w2*t2 + … で正規化・合算
似た構図はクラスタリング後、各クラスから代表を残す | perceptual hash で類似判定 → さらに CLIP 埋め込みで refine
選別結果は LR に星／カラーラベルで反映 | XMP サイドカー の <xmp:Rating> を書いて Classic へ再読み込み


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
$ python selector_pipeline.py --preview-dir "スマートプレビューのlrdataのディレクトリ" --output-dir ./xmp_out \
      --clusters 50 --topk 3

‑‑preview-dir  : Lightroom で生成したプレビューファイル群（JPEG）のルート
‑‑output-dir   : レーティングを書き込んだ XMP を吐き出す場所
‑‑clusters     : KMeans のクラスタ数（お好みで）
‑‑topk         : 各クラスタからピックする枚数
‑‑weights      : スコア重み JSON 文字列 (optional)