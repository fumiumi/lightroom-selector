# lightroom-selector
撮り溜めた写真をMLモデルに自動で選別させて、Lightroomの星とフラグを自動でつけるプラグイン

## 使い方

lr_generate_rating_json.py 
====================
Lightroom スマートプレビュー（dng, ~2048px）を対象に
1. 類似構図クラスタリング
2. 主題・構図・露出スコアリング
3. クラスタ内で上位カットを選定
4. XMP サイドカーに ★レーティングを書き込み
までを一気通貫で実行するスケルトン。

### コマンド

```bash
python lr_generate_rating_json.py 
--preview-dir "スマートプレビューのlrdataのディレクトリ" \
--output-dir ./xmp_out \
--clusters 50 --topk 3
```

‑‑preview-dir  : Lightroom で生成したプレビューファイル群（JPEG）のルート
‑‑output-dir   : レーティングを書き込んだ XMP を吐き出す場所
‑‑clusters     : KMeans のクラスタ数（お好みで）
‑‑topk         : 各クラスタからピックする枚数
‑‑weights      : スコア重み JSON 文字列 (optional)