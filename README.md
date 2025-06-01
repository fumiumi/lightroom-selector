# lightroom-selector
撮り溜めた写真をMLモデルに自動で選別させて、Lightroomの星とフラグを自動でつけるプラグイン

## 使い方

### 使用するPythonバージョン

3.11を使用する。スクリプトで使用しているtensorflowが3.13をサポートしていないため。

プロジェクト内にPython 3.11 で仮想環境を作成し、仮想環境をアクティベート

```bash
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 必要なライブラリをインストール

```bash
pip install opencv-python pillow imagehash numpy scikit-learn torch ultralytics piexif rich tensorflow
```

### コマンド

```bash
py -3.11 lr_generate_rating_json.py 
--preview-dir "スマートプレビューのlrdataのディレクトリ" \
--output-dir ./json_out \
--clusters 50 --topk 3
```

‑‑preview-dir  : Lightroom で生成したプレビューファイル群（JPEG）のルート
‑‑output-dir   : レーティングを書き込んだ XMP を吐き出す場所
‑‑clusters     : KMeans のクラスタ数（お好みで）
‑‑topk         : 各クラスタからピックする枚数
‑‑weights      : スコア重み JSON 文字列 (optional)