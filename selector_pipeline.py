#!/usr/bin/env python3
"""
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

step‑by‑step の詳細は README.md or チャット本文を参照！
"""

from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np
from PIL import Image
import imagehash
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import torch
from ultralytics import YOLO  # yolov8
# TODO: piexif で XMP 書込み or ExifTool ラッパ

# ========================= Utility ============================

def load_image(path: Path) -> np.ndarray:
    """OpenCV 形式で画像読み込み (BGR). DNG は rawpy で処理"""
    ext = path.suffix.lower()
    # Raw (.dng) は rawpy で現像
    if ext == ".dng":
        try:
            import rawpy
            # RawPy で現像
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess()
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except ImportError:
            raise RuntimeError("rawpy が見つかりません: pip install rawpy")
        except Exception as e:
            # rawpy がサポートしないスマートプレビュー等の場合は PIL でフォールバック
            try:
                img_pil = Image.open(str(path))
                rgb = np.array(img_pil)
                return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception:
                raise RuntimeError(f"Failed to open DNG (rawpy, PIL 共にNG): {path}") from e

    # JPEG 等は OpenCV で読み込み
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Failed to open {path}")
    return img

# ---- Histogram / Exposure -----------------------------------

def exposure_score(img: np.ndarray) -> float:
    """クリップ率から露出バランスを 0-1 で返す"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    total = hist.sum()
    clip_white = hist[250:].sum() / total
    clip_black = hist[:5].sum() / total
    score = 1.0 - min(1.0, clip_white + clip_black)  # 0 (bad)‑1 (good)
    return float(score)

# ---- Composition --------------------------------------------

def thirds_alignment_score(bbox, img_shape) -> float:
    """被写体 bbox と三分割交点の距離で 0-1."""
    h, w = img_shape[:2]
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    thirds_x = [w / 3, 2 * w / 3]
    thirds_y = [h / 3, 2 * h / 3]
    dists = [np.hypot(cx - tx, cy - ty) for tx in thirds_x for ty in thirds_y]
    norm_dist = min(dists) / np.hypot(w, h)
    return 1.0 - norm_dist  # 1 が中心に近い

# ---- Subject Detection ---------------------------------------

yolo = YOLO("yolov8n.pt")  # 軽量モデル。初回 DL に数百 MB


def detect_subject(img: np.ndarray):
    """YOLOv8 で最大スコアの bbox を返す (xyxy)."""
    res = yolo.predict(source=img, imgsz=640, conf=0.25, verbose=False)
    boxes = res[0].boxes
    if boxes.shape[0] == 0:
        return None
    # 最大面積 or 最大 conf で決定
    idx = torch.argmax(boxes.conf).item()
    return boxes.xyxy[idx].cpu().numpy()

# ---- Perceptual Hash for clustering --------------------------

def hash_vector(path: Path) -> np.ndarray:
    """pHash を 64-bit -> 64-dim 0/1 ベクトル化"""
    img = Image.open(path)
    ph = imagehash.phash(img, hash_size=8)  # 64 bits
    # convert to vector of 0/1
    v = np.array([int(b) for b in bin(int(str(ph), 16))[2:].zfill(64)], dtype=np.float32)
    return v

# ========================= Pipeline ===========================

class PhotoScore:
    def __init__(self, path: Path):
        self.path = path
        self.img = load_image(path)
        self.scores: Dict[str, float] = {}
        self.total: float = 0.0
        self.bbox = None

    def evaluate(self, weights: Dict[str, float]):
        # 主題検出
        self.bbox = detect_subject(self.img)
        focus_score = 1.0 if self.bbox is not None else 0.0
        # 構図
        comp_score = thirds_alignment_score(self.bbox, self.img.shape) if self.bbox is not None else 0.0
        # 露出
        exp_score = exposure_score(self.img)
        # 合算
        self.scores = {
            "composition": comp_score,
            "focus": focus_score,
            "thirds": comp_score,  # 同一視点 例
            "exposure": exp_score,
        }
        self.total = (
            weights["tier1"] * self.scores["composition"] +
            weights["tier2"] * self.scores["focus"] +
            weights["tier3"] * self.scores["thirds"] +
            weights["tier4"] * self.scores["exposure"]
        )


# ========================= Main ===============================

def main():
    parser = argparse.ArgumentParser(description="Auto-select best photos from Lightroom previews")
    parser.add_argument("--preview-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--clusters", type=int, default=50)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--weights", type=str, default="{\"tier1\":0.4,\"tier2\":0.3,\"tier3\":0.2,\"tier4\":0.1}")
    args = parser.parse_args()

    weights = json.loads(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 読み込み & スコア計算
    photos: List[PhotoScore] = []
    for p in args.preview_dir.rglob("*.dng"): # Lightroom のプレビューファイル
        ps = PhotoScore(p)
        ps.evaluate(weights)
        photos.append(ps)

    # 2. ハッシュベクトルでクラスタリング
    vecs = np.stack([hash_vector(p.path) for p in photos])
    km = KMeans(n_clusters=args.clusters, random_state=0)
    labels = km.fit_predict(vecs)

    # 3. クラスタ毎に top‑k
    clusters: Dict[int, List[PhotoScore]] = {}
    for ps, lbl in zip(photos, labels):
        clusters.setdefault(lbl, []).append(ps)

    selected: List[PhotoScore] = []
    for lbl, plist in clusters.items():
        plist.sort(key=lambda x: x.total, reverse=True)
        selected.extend(plist[:args.topk])

    # 4. XMP サイドカーに評価を書き込み
    for ps in selected:
        print(f"★ SELECTED {ps.path.name} score={ps.total:.2f}")
        # --------------------------------------------------
        # ● XMPサイドカーの生成
        # Lightroom は同名 .xmp を横に置くことで評価を読み込みます
        rating = 5
        xmp_path = args.output_dir / f"{ps.path.stem}.xmp"
        # XMP のヘッダー／RDF 部分に Rating 属性を埋め込む
        xmp_content = f'''<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>
<x:xmpmeta xmlns:x="adobe:ns:meta/">
  <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
    <rdf:Description rdf:about=""
      xmlns:xmp="http://ns.adobe.com/xap/1.0/"
      xmp:Rating="{rating}"/>
  </rdf:RDF>
</x:xmpmeta>
<?xpacket end="w"?>'''
        # ファイルへ書き出し
        xmp_path.write_text(xmp_content, encoding="utf-8")
        print(f"  -> Wrote XMP: {xmp_path}")

    print(f"\nFinished. Selected {len(selected)} / {len(photos)} shots.")


if __name__ == "__main__":
    main()
