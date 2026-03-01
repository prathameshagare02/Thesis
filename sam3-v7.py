from __future__ import annotations

from typing import Dict, Tuple, Optional
import argparse
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------- Optional deps -----------------------------
try:
    import torch

    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

try:
    import open_clip
    from PIL import Image

    OPENCLIP_AVAILABLE = True
except Exception:
    OPENCLIP_AVAILABLE = False


# ----------------------------- File Picker -----------------------------
def _get_tk_root():
    try:
        import tkinter as tk

        root = tk.Tk()
        root.withdraw()
        try:
            root.attributes("-topmost", True)
        except Exception:
            pass
        return root
    except ImportError as e:
        raise RuntimeError("tkinter not available.") from e


def pick_image_file(
    title: str = "Select an image",
    filetypes=(
        ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
        ("All files", "*.*"),
    ),
) -> str:
    from tkinter import filedialog

    root = _get_tk_root()
    path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    if not path:
        raise ValueError("No file selected.")
    return path


# ----------------------------- Utils -----------------------------
def _l2_normalize(x: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    return x / (n + eps)


def _cosine_sim_map(feat_hw_c: np.ndarray, text_c: np.ndarray) -> np.ndarray:
    """
    feat_hw_c: (H,W,C) L2-normalized
    text_c:    (C,)   L2-normalized
    returns:   (H,W)
    """
    return (feat_hw_c * text_c[None, None, :]).sum(axis=-1)


def _kmeans_2class(
    X: np.ndarray,
    init_labels: np.ndarray,
    max_iter: int = 20,
    tol: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simple 2-class k-means.
    X: (N,C) float32
    init_labels: (N,) {0,1}
    returns: labels (N,), centroids (2,C)
    """
    labels = init_labels.astype(np.int32).copy()
    N, C = X.shape

    def centroid_for(k: int) -> np.ndarray:
        idx = labels == k
        if not np.any(idx):
            ridx = np.random.randint(0, N)
            return X[ridx].copy()
        return X[idx].mean(axis=0)

    c0 = centroid_for(0)
    c1 = centroid_for(1)
    centroids = np.stack([c0, c1], axis=0)

    prev_inertia = None
    for _ in range(max_iter):
        d0 = ((X - centroids[0]) ** 2).sum(axis=1)
        d1 = ((X - centroids[1]) ** 2).sum(axis=1)
        new_labels = (d1 < d0).astype(np.int32)

        labels = new_labels
        c0_new = centroid_for(0)
        c1_new = centroid_for(1)
        centroids_new = np.stack([c0_new, c1_new], axis=0)

        inertia = float(np.minimum(d0, d1).sum())
        if prev_inertia is not None and abs(prev_inertia - inertia) / (prev_inertia + 1e-9) < tol:
            centroids = centroids_new
            break
        prev_inertia = inertia
        centroids = centroids_new

    return labels, centroids


# ----------------------------- Core Detector -----------------------------
class ClipKMeansRustDetector:
    """
    Pipeline (OpenCLIP text+image embeddings):

      1) Load OpenCLIP model.
      2) Build TWO text embeddings:
           - "rusty metal"
           - "clean shining metal"
      3) Extract per-pixel feature vectors from CLIP ViT patch embeddings.
      4) Run k-means (k=2) over pixel features to classify METAL vs NON-METAL.
         - Initialize labels using similarity(rusty) vs similarity(clean).
         - Decide which cluster is METAL by higher average "metal-likeness"
           (max(sim_rust, sim_clean)).
      5) Inside METAL pixels, classify RUST vs CLEAN using sim_rust > sim_clean.
      6) Report:
           - metal_coverage (% of image)
           - rust_coverage_within_metal (% of metal that is rusty)
           - rust_coverage_image (% of whole image)
    """

    def __init__(
        self,
        device: str = "",
        verbose: bool = True,
        kmeans_max_iter: int = 20,
        viz_downscale_max_side: int = 900,
        clip_model: str = "ViT-B-32",
        clip_pretrained: str = "openai",
    ):
        self.verbose = bool(verbose)
        self.kmeans_max_iter = int(kmeans_max_iter)
        self.viz_downscale_max_side = int(viz_downscale_max_side)

        self.prompt_rust = "rusty metal"
        self.prompt_clean = "clean shining metal"

        self.device = device or ("cuda" if TORCH_AVAILABLE and torch.cuda.is_available() else "cpu")
        self.clip_model_name = clip_model
        self.clip_pretrained = clip_pretrained

        self.model = None
        self.preprocess = None

        if self.verbose:
            print("ClipKMeansRustDetector initialized:")
            print(f"  Prompts: {self.prompt_rust!r} vs {self.prompt_clean!r}")
            print(f"  k-means max_iter: {self.kmeans_max_iter}")
            print(f"  device: {self.device}")
            print(f"  CLIP model: {self.clip_model_name} ({self.clip_pretrained})")

        self.load_clip_model()

    def _log(self, msg: str):
        if self.verbose:
            print(f"  → {msg}")

    # ----------------------------- CLIP loading -----------------------------
    def load_clip_model(self):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required (import torch failed).")
        if not OPENCLIP_AVAILABLE:
            raise RuntimeError("open_clip is required. Install: pip install open_clip_torch")

        self._log("Loading OpenCLIP model...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            self.clip_model_name, pretrained=self.clip_pretrained
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        self._log("OpenCLIP loaded.")

    # ----------------------------- Embeddings extraction -----------------------------
    def _encode_text_clip(self, prompts: list[str]) -> np.ndarray:
        tokens = open_clip.tokenize(prompts).to(self.device)
        with torch.no_grad():
            text = self.model.encode_text(tokens)
        text = text.detach().float().cpu().numpy().astype(np.float32)
        return _l2_normalize(text, axis=1)

    def _image_feature_map_clip(self, image_bgr: np.ndarray) -> np.ndarray:
        if self.model is None or self.preprocess is None:
            raise RuntimeError("CLIP model not loaded.")

        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(image_rgb)
        image_t = self.preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            visual = self.model.visual
            if hasattr(visual, "forward_features"):
                feats = visual.forward_features(image_t)
            else:
                raise RuntimeError("CLIP visual encoder does not expose forward_features().")

            if feats.ndim != 3:
                raise RuntimeError("Unexpected CLIP visual features shape.")

            if feats.shape[1] > 1:
                feats = feats[:, 1:, :]

            if hasattr(visual, "grid_size"):
                gh, gw = visual.grid_size
            else:
                n = feats.shape[1]
                g = int(np.sqrt(n))
                gh, gw = g, g

            feat_map = feats[0].reshape(gh, gw, -1).contiguous()

        feat_map = feat_map.detach().float().cpu().numpy().astype(np.float32)
        feat_map = _l2_normalize(feat_map, axis=2)

        H, W = image_bgr.shape[:2]
        feat_map_t = torch.from_numpy(feat_map).permute(2, 0, 1).unsqueeze(0)
        feat_up = torch.nn.functional.interpolate(
            feat_map_t, size=(H, W), mode="bilinear", align_corners=False
        )
        feat_up = feat_up[0].permute(1, 2, 0).contiguous().numpy().astype(np.float32)
        feat_up = _l2_normalize(feat_up, axis=2)

        return feat_up

    # ----------------------------- Main analysis -----------------------------
    def analyze(self, image_path: str) -> Dict:
        t0 = time.time()
        self._log(f"Loading image: {image_path}")
        original = cv2.imread(image_path)
        if original is None:
            raise ValueError(f"Image not found: {image_path}")
        H, W = original.shape[:2]
        t_load = time.time() - t0

        t0 = time.time()
        self._log("Encoding text prompts with OpenCLIP...")
        text_emb = self._encode_text_clip([self.prompt_rust, self.prompt_clean])
        rust_text = text_emb[0]
        clean_text = text_emb[1]
        t_text = time.time() - t0

        t0 = time.time()
        self._log("Extracting CLIP per-pixel feature map...")
        feat_hw_c = self._image_feature_map_clip(original)
        t_feat = time.time() - t0

        t0 = time.time()
        self._log("Computing similarity maps...")
        sim_rust = _cosine_sim_map(feat_hw_c, rust_text)
        sim_clean = _cosine_sim_map(feat_hw_c, clean_text)
        metal_likeness = np.maximum(sim_rust, sim_clean)
        t_sim = time.time() - t0

        t0 = time.time()
        self._log("Running k-means (k=2) on pixel features for METAL vs NON-METAL...")

        X = feat_hw_c.reshape(-1, feat_hw_c.shape[2]).astype(np.float32)
        init_labels = (sim_clean.reshape(-1) > sim_rust.reshape(-1)).astype(np.int32)

        labels, _ = _kmeans_2class(X, init_labels=init_labels, max_iter=self.kmeans_max_iter)
        labels_hw = labels.reshape(H, W)

        ml0 = float(metal_likeness[labels_hw == 0].mean()) if np.any(labels_hw == 0) else -1e9
        ml1 = float(metal_likeness[labels_hw == 1].mean()) if np.any(labels_hw == 1) else -1e9
        metal_cluster = 0 if ml0 >= ml1 else 1

        metal_mask = (labels_hw == metal_cluster).astype(np.uint8)
        t_kmeans = time.time() - t0

        rust_mask = (metal_mask == 1) & (sim_rust > sim_clean)
        rust_mask_u8 = rust_mask.astype(np.uint8)

        metal_pixels = int(metal_mask.sum())
        rust_pixels = int(rust_mask_u8.sum())
        total_pixels = H * W

        metal_coverage = 100.0 * metal_pixels / max(1, total_pixels)
        rust_within_metal = 100.0 * rust_pixels / max(1, metal_pixels)
        rust_in_image = 100.0 * rust_pixels / max(1, total_pixels)

        self._log(
            f"metal_coverage={metal_coverage:.2f}% | "
            f"rust_within_metal={rust_within_metal:.2f}% | "
            f"rust_in_image={rust_in_image:.2f}%"
        )

        return {
            "original": original,
            "feat_hw_c": feat_hw_c,
            "sim_rust": sim_rust,
            "sim_clean": sim_clean,
            "metal_likeness": metal_likeness,
            "metal_mask": metal_mask,
            "rust_mask": rust_mask_u8,
            "metal_coverage_pct": metal_coverage,
            "rust_within_metal_pct": rust_within_metal,
            "rust_in_image_pct": rust_in_image,
            "timings": {
                "load": t_load,
                "text_encode": t_text,
                "feature_map": t_feat,
                "similarity": t_sim,
                "kmeans": t_kmeans,
                "total": t_load + t_text + t_feat + t_sim + t_kmeans,
            },
            "prompts": {
                "rust": self.prompt_rust,
                "clean": self.prompt_clean,
            },
        }

    # ----------------------------- Visualization -----------------------------
    def visualize(self, results: Dict, save_path: Optional[str] = None) -> plt.Figure:
        original = results["original"]
        metal_mask = results["metal_mask"]
        rust_mask = results["rust_mask"]
        sim_rust = results["sim_rust"]
        sim_clean = results["sim_clean"]

        H, W = original.shape[:2]

        scale = 1.0
        mx = max(H, W)
        if mx > self.viz_downscale_max_side:
            scale = self.viz_downscale_max_side / float(mx)

        def resize_img(img):
            if scale == 1.0:
                return img
            nh = max(1, int(img.shape[0] * scale))
            nw = max(1, int(img.shape[1] * scale))
            return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

        def resize_mask(msk):
            if scale == 1.0:
                return msk
            nh = max(1, int(msk.shape[0] * scale))
            nw = max(1, int(msk.shape[1] * scale))
            return cv2.resize(msk.astype(np.uint8), (nw, nh), interpolation=cv2.INTER_NEAREST)

        vis_orig = resize_img(original)
        vis_metal = resize_mask(metal_mask)
        vis_rust = resize_mask(rust_mask)

        metal_overlay = vis_orig.copy()
        tmp = metal_overlay.copy()
        tmp[vis_metal == 1] = (255, 0, 0)
        metal_overlay = cv2.addWeighted(metal_overlay, 0.65, tmp, 0.35, 0)

        rust_overlay = vis_orig.copy()
        tmp2 = rust_overlay.copy()
        tmp2[vis_rust == 1] = (0, 0, 255)
        rust_overlay = cv2.addWeighted(rust_overlay, 0.65, tmp2, 0.35, 0)

        sim_r = resize_img(sim_rust.astype(np.float32))
        sim_c = resize_img(sim_clean.astype(np.float32))

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "OpenCLIP Embeddings + k-means Metal/Non-Metal + Rust/Clean\n"
            f"metal={results['metal_coverage_pct']:.1f}% | "
            f"rust within metal={results['rust_within_metal_pct']:.1f}% | "
            f"rust in image={results['rust_in_image_pct']:.1f}%",
            fontsize=12,
        )

        axes[0, 0].imshow(cv2.cvtColor(vis_orig, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title("1) Input")
        axes[0, 0].axis("off")

        axes[0, 1].imshow(cv2.cvtColor(metal_overlay, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title("2) Metal mask overlay (k-means)")
        axes[0, 1].axis("off")

        axes[0, 2].imshow(cv2.cvtColor(rust_overlay, cv2.COLOR_BGR2RGB))
        axes[0, 2].set_title("3) Rust mask overlay (sim_rust > sim_clean inside metal)")
        axes[0, 2].axis("off")

        axes[1, 0].imshow(sim_r, cmap="viridis")
        axes[1, 0].set_title(f"4) sim('{results['prompts']['rust']}')")
        axes[1, 0].axis("off")

        axes[1, 1].imshow(sim_c, cmap="viridis")
        axes[1, 1].set_title(f"5) sim('{results['prompts']['clean']}')")
        axes[1, 1].axis("off")

        axes[1, 2].imshow(vis_rust, cmap="gray")
        axes[1, 2].set_title("6) Rust mask")
        axes[1, 2].axis("off")

        plt.tight_layout()
        plt.show()

        if save_path:
            out_dir = os.path.dirname(save_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            fig.savefig(save_path, dpi=300)
            self._log(f"Saved visualization to: {save_path}")

        return fig


# ----------------------------- CLI + Main -----------------------------
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Rust detection with OpenCLIP embeddings + k-means metal/non-metal + rust/clean text similarity"
    )
    p.add_argument("--image", type=str, default="", help="Path to image; if empty, file dialog opens.")
    p.add_argument("--device", type=str, default="", help="Device override: ''=auto, 'cpu', 'cuda'.")
    p.add_argument("--kmeans_max_iter", type=int, default=20)
    p.add_argument("--res_dir", type=str, default="ClipKMeans")
    p.add_argument("--no_viz", action="store_true", help="Disable visualization.")
    p.add_argument("--clip_model", type=str, default="ViT-B-32")
    p.add_argument("--clip_pretrained", type=str, default="openai")
    return p


def main():
    args = build_argparser().parse_args()

    if args.image:
        if not os.path.exists(args.image):
            raise SystemExit(f"--image does not exist: {args.image}")
        image_path = args.image
    else:
        image_path = pick_image_file()

    detector = ClipKMeansRustDetector(
        device=args.device,
        verbose=True,
        kmeans_max_iter=args.kmeans_max_iter,
        clip_model=args.clip_model,
        clip_pretrained=args.clip_pretrained,
    )

    results = detector.analyze(image_path)

    print("\n=== FINAL RESULTS ===")
    print(f"Metal coverage (% of image):        {results['metal_coverage_pct']:.2f}%")
    print(f"Rust coverage (% within metal):     {results['rust_within_metal_pct']:.2f}%")
    print(f"Rust coverage (% of whole image):   {results['rust_in_image_pct']:.2f}%")
    print(f"Timings: {results['timings']}")

    if not args.no_viz:
        out = f"results/{args.res_dir}/{os.path.splitext(os.path.basename(image_path))[0]}_clip_kmeans.png"
        detector.visualize(results, save_path=out)


if __name__ == "__main__":
    main()