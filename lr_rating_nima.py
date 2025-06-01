#!/usr/bin/env python3
"""
lr_generate_rating_json.py (v3)
================================
• YOLO で主題検出
• NIMA で美的スコア (複数 URL フォールバック + ローカル .h5)
• 三分割距離・露出・クラスタリング
• ratings.json を出力 (Lua Plug‑in へ渡す)

改善点 v3
----------
1. **NumPy 2.x 非対応回避**: 用いる PyTorch 拡張が NumPy 2.x 未サポートのため、起動時に NumPy バージョンを確認し、<2.0.0 でない場合はエラーを通知。
2. **CUDA 自動フォールバック**: “auto” モードで GPU が使えなければ CPU を利用。
3. **ログ & エラー処理強化**
4. **`--preview-ext` で拡張子を指定可**
5. **`--device` CPU/GPU 切替 (YOLO+TF) を CLI で制御**
6. **tqdm で進捗可視化**
"""

from __future__ import annotations
import argparse, json, logging, sys
from pathlib import Path
from typing import Dict, List

import cv2
from PIL import Image
import numpy as np
import imagehash
import torch
from ultralytics import YOLO
from sklearn.cluster import KMeans
from tqdm import tqdm

# ------------ Logging ----------------
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger("LR-NIMA")

# ---------- NumPy Version Check ----------
try:
    import numpy as _np_check
    from packaging import version as _ver
    if _ver.parse(_np_check.__version__) >= _ver.parse("2.0.0"):
        log.error("Detected NumPy %s which is incompatible. Please install numpy<2.0.0", _np_check.__version__)
        sys.exit(1)
except ImportError:
    log.error("NumPy is not installed.")
    sys.exit(1)

# ---------- Torch & CUDA Setup ----------
def choose_device(requested: str) -> str:
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA requested but unavailable. Falling back to CPU.")
        return "cpu"
    return requested

# ---------- Image Loading ----------

def load_image(path: Path) -> np.ndarray:
    """OpenCV BGR で画像ロード。DNG は rawpy 現像 + PIL フォールバック"""
    ext = path.suffix.lower()
    if ext == ".dng":
        try:
            import rawpy
            with rawpy.imread(str(path)) as raw:
                rgb = raw.postprocess(no_auto_bright=True, output_bps=8)
            return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        except Exception:
            log.warning("rawpy failed for %s, using PIL fallback", path.name)
            img = Image.open(str(path))
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img = cv2.imread(str(path))
    if img is None:
        raise RuntimeError(f"Cannot open image: {path}")
    return img

# ---------- Exposure ----------

def exposure_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    clips = (hist[:5].sum() + hist[250:].sum()) / hist.sum()
    return float(1.0 - min(1.0, clips))

# ---------- Thirds ----------

def thirds_score(bbox, shape) -> float:
    if bbox is None:
        return 0.0
    h, w = shape[:2]
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    points = [(w/3, h/3), (2*w/3, h/3), (w/3, 2*h/3), (2*w/3, 2*h/3)]
    d = min([np.hypot(cx - x, cy - y) for x, y in points]) / np.hypot(w, h)
    return 1.0 - d

# ---------- YOLO Subject Detection ----------
_yolo_cache: Dict[str, YOLO] = {}

def get_yolo(model_path: str, device: str) -> YOLO:
    if device not in _yolo_cache:
        model = YOLO(model_path, task="detect")
        model.to(device)
        _yolo_cache[device] = model
    return _yolo_cache[device]

def detect_subject(img: np.ndarray, device: str) -> list | None:
    mdl = get_yolo("yolov8n.pt", device)
    results = mdl.predict(img, verbose=False, device=device, conf=0.25, imgsz=640)[0]
    boxes = results.boxes
    if len(boxes) == 0:
        return None
    idx = int(torch.argmax(boxes.conf))
    return boxes.xyxy[idx].cpu().numpy()

# ---------- NIMA Aesthetic Score ----------
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

NIMA_CANDIDATES = [
    "https://tfhub.dev/google/nima/mobilenet_v2/2",
    "https://tfhub.dev/google/nima/mobilenet_v2/1",
    "https://tfhub.dev/google/nima_mobilenet_v2/1"
]
LOCAL_H5 = "nima_mobilenet_weights.h5"
_nima = None

def load_nima_model() -> tf.keras.Model:
    global _nima
    if _nima is not None:
        return _nima
    # ① TF-Hub URLs
    for url in NIMA_CANDIDATES:
        try:
            log.info("Trying NIMA from %s", url)
            layer = hub.KerasLayer(url, trainable=False)
            model = tf.keras.Sequential([layer])
            # Dry run
            dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
            _ = model(dummy)
            log.info("✔ loaded NIMA from %s", url)
            _nima = model
            return _nima
        except Exception as e:
            log.warning("NIMA load failed @ %s: %s", url, e.__class__.__name__)
    # ② ローカル .h5 重み
    if Path(LOCAL_H5).exists():
        base = tf.keras.applications.MobileNetV2(include_top=False, input_shape=(224, 224, 3))
        x = tf.keras.layers.GlobalAveragePooling2D()(base.output)
        out = tf.keras.layers.Dense(10, activation="softmax")(x)
        model = tf.keras.Model(base.input, out)
        model.load_weights(LOCAL_H5)
        log.info("✔ loaded local NIMA weights from %s", LOCAL_H5)
        _nima = model
        return _nima
    # ③ 失敗
    raise RuntimeError(f"NIMA weights not found. Provide {LOCAL_H5} or check URLs.")

def nima_score(img: np.ndarray) -> float:
    model = load_nima_model()
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    arr = preprocess_input(rgb.astype(np.float32))[None] / 255.0
    probs = model(arr)[0].numpy()
    # 1-10 -> 0-1 正規化
    return float(np.sum(probs * np.arange(1, 11)) / 10.0)

# ---------- Perceptual Hash for Clustering ----------

def hash_vec(path: Path) -> np.ndarray:
    ph = imagehash.phash(Image.open(path), hash_size=8)
    bits = np.array(list(map(int, bin(int(str(ph), 16))[2:].zfill(64))), dtype=np.float32)
    return bits

# ---------- PhotoScore Class ----------
class PhotoScore:
    def __init__(self, path: Path, device: str):
        self.path = path
        self.img = load_image(path)
        self.device = device
        self.total: float = 0.0

    def evaluate(self, w: Dict[str, float]):
        bbox = detect_subject(self.img, self.device)
        comp = thirds_score(bbox, self.img.shape)
        focus = float(bbox is not None)
        expos = exposure_score(self.img)
        aest = nima_score(self.img)
        self.total = (
            w.get("tier1", 0) * comp +
            w.get("tier2", 0) * aest +
            w.get("tier3", 0) * focus +
            w.get("tier4", 0) * comp +
            w.get("tier5", 0) * expos
        )

# ---------- Main Pipeline ----------

def main():
    ap = argparse.ArgumentParser(description="Auto-select best photos from Lightroom previews")
    ap.add_argument("--preview-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument("--preview-ext", default=".dng")
    ap.add_argument("--clusters", type=int, default=50)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument(
        "--weights", default='{"tier1":0.3,"tier2":0.3,"tier3":0.2,"tier4":0.1,"tier5":0.1}'
    )
    ap.add_argument("--device", choices=["cpu", "cuda", "auto"], default="auto")
    args = ap.parse_args()

    device = choose_device(args.device)
    log.info("Using device: %s", device)

    w = json.loads(args.weights)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    photos: List[PhotoScore] = []
    files = list(args.preview_dir.rglob(f"*{args.preview_ext}"))
    for p in tqdm(files, desc="Scoring"):
        try:
            ps = PhotoScore(p, device)
            ps.evaluate(w)
            photos.append(ps)
        except Exception as e:
            log.warning("Failed to score %s: %s", p.name, e)

    if not photos:
        log.error("No photos scored. Exiting.")
        sys.exit(1)

    vecs = np.stack([hash_vec(ps.path) for ps in photos])
    n_clusters = min(args.clusters, len(photos))
    lbls = KMeans(n_clusters=n_clusters, random_state=0).fit_predict(vecs)

    cluster_map: Dict[int, List[PhotoScore]] = {}
    for ps, lbl in zip(photos, lbls):
        cluster_map.setdefault(lbl, []).append(ps)

    selected: List[PhotoScore] = []
    for pls in cluster_map.values():
        pls.sort(key=lambda x: x.total, reverse=True)
        selected.extend(pls[: args.topk])

    # dynamic threshold (上位25% を★5、それ以外を★4とする例)
    totals = np.array([ps.total for ps in selected])
    q75 = np.quantile(totals, 0.75) if len(totals) > 0 else 0.0

    out = []
    for ps in selected:
        rating = 5 if ps.total >= q75 else 4
        out.append({"file": ps.path.name, "score": round(ps.total, 4), "rating": rating})

    json_path = args.output_dir / "ratings.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    log.info("Wrote %s", json_path)

if __name__ == "__main__":
    main()
