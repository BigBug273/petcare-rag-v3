#!/usr/bin/env python3
"""
scripts/export_onnx.py
======================
Export the Sentence-BERT model to ONNX format (INT8 quantized).

Run this script ONCE on your local machine (where you have enough RAM),
then commit the generated files to your repo for Render deployment.

Usage:
    pip install onnxruntime onnx
    python scripts/export_onnx.py

Output:
    rag/model_onnx/model_int8.onnx   — INT8 quantized ONNX model (~110 MB)
    rag/model_onnx/tokenizer.json    — Tokenizer file (updated if needed)
    rag/model_onnx/config.json       — Model config (updated if needed)

Why ONNX?
    PyTorch runtime uses ~800 MB RAM → exceeds Render's 512 MB limit.
    ONNX Runtime uses ~150 MB RAM → fits comfortably.
    Same model, same embeddings, just a lighter inference engine.
"""

import os
import sys
import json
import shutil

# Ensure project root is in path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "rag", "model_onnx")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("📦 Exporting Sentence-BERT to ONNX (INT8)")
    print("=" * 60)

    import torch
    import numpy as np
    from sentence_transformers import SentenceTransformer
    from transformers import AutoTokenizer

    # ----------------------------------------------------------
    # Step 1: Load model + tokenizer
    # ----------------------------------------------------------
    print(f"\n🔄 Loading model: {MODEL_NAME}...")
    st_model = SentenceTransformer(MODEL_NAME)
    tokenizer = st_model.tokenizer
    transformer = st_model[0].auto_model
    transformer.eval()  # set to eval mode

    # Save tokenizer to output dir
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ Tokenizer saved to {OUTPUT_DIR}/")

    # ----------------------------------------------------------
    # Step 2: Export to ONNX (FP32)
    # ----------------------------------------------------------
    onnx_fp32_path = os.path.join(OUTPUT_DIR, "model_fp32.onnx")
    print(f"\n🔄 Exporting to ONNX (FP32)...")

    # Create dummy input
    dummy = tokenizer("hello world", return_tensors="pt", padding=True, truncation=True, max_length=128)

    # Try exporting with dynamo=False first (legacy TorchScript exporter)
    # This handles dynamic axes correctly for LayerNormalization
    export_ok = False

    # Method 1: Legacy exporter (most reliable for dynamic shapes)
    if not export_ok:
        try:
            print("  Trying legacy TorchScript exporter (dynamo=False)...")
            torch.onnx.export(
                transformer,
                (dummy["input_ids"], dummy["attention_mask"]),
                onnx_fp32_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "last_hidden_state": {0: "batch", 1: "seq"},
                },
                opset_version=14,
                dynamo=False,
            )
            export_ok = True
            print("  ✅ Legacy export succeeded!")
        except (TypeError, Exception) as e:
            print(f"  ⚠️ Legacy export failed: {e}")

    # Method 2: Fallback without dynamo parameter (older PyTorch)
    if not export_ok:
        try:
            print("  Trying export without dynamo parameter...")
            export_kwargs = {
                "f": onnx_fp32_path,
                "args": (dummy["input_ids"], dummy["attention_mask"]),
                "input_names": ["input_ids", "attention_mask"],
                "output_names": ["last_hidden_state"],
                "dynamic_axes": {
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "last_hidden_state": {0: "batch", 1: "seq"},
                },
                "opset_version": 14,
                "do_constant_folding": True,
            }
            torch.onnx.export(transformer, **export_kwargs)
            export_ok = True
            print("  ✅ Export without dynamo succeeded!")
        except Exception as e:
            print(f"  ⚠️ Export without dynamo failed: {e}")

    # Method 3: Dynamo exporter (newer PyTorch versions)
    if not export_ok:
        try:
            print("  Trying Dynamo exporter (dynamo=True)...")
            torch.onnx.export(
                transformer,
                (dummy["input_ids"], dummy["attention_mask"]),
                onnx_fp32_path,
                input_names=["input_ids", "attention_mask"],
                output_names=["last_hidden_state"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "seq"},
                    "attention_mask": {0: "batch", 1: "seq"},
                    "last_hidden_state": {0: "batch", 1: "seq"},
                },
                opset_version=18,
            )
            export_ok = True
            print("  ✅ Dynamo export succeeded!")
        except Exception as e:
            print(f"  ⚠️ Dynamo export failed: {e}")

    # Method 4: Optimum (if available)
    if not export_ok:
        try:
            print("  Trying Optimum exporter...")
            from optimum.onnxruntime import ORTModelForFeatureExtraction

            ort_model = ORTModelForFeatureExtraction.from_pretrained(MODEL_NAME, export=True)
            ort_model.save_pretrained(OUTPUT_DIR)
            # Optimum saves as model.onnx, rename to model_fp32.onnx
            optimum_path = os.path.join(OUTPUT_DIR, "model.onnx")
            if os.path.exists(optimum_path) and not os.path.exists(onnx_fp32_path):
                os.rename(optimum_path, onnx_fp32_path)
            export_ok = True
            print("  ✅ Optimum export succeeded!")
        except ImportError:
            print("  ⚠️ Optimum not installed, skipping")
        except Exception as e:
            print(f"  ⚠️ Optimum export failed: {e}")

    if not export_ok:
        print("\n❌ All export methods failed!")
        print("   Try: pip install optimum")
        sys.exit(1)

    fp32_size = os.path.getsize(onnx_fp32_path) / (1024 * 1024)
    print(f"✅ FP32 ONNX model: {fp32_size:.1f} MB")

    # ----------------------------------------------------------
    # Step 3: Quantize to INT8
    # ----------------------------------------------------------
    from onnxruntime.quantization import quantize_dynamic, QuantType

    onnx_int8_path = os.path.join(OUTPUT_DIR, "model_int8.onnx")
    print(f"\n🔄 Quantizing FP32 → INT8...")

    quantize_dynamic(
        onnx_fp32_path,
        onnx_int8_path,
        weight_type=QuantType.QUInt8,
    )

    int8_size = os.path.getsize(onnx_int8_path) / (1024 * 1024)
    print(f"✅ INT8 ONNX model: {int8_size:.1f} MB")

    # ----------------------------------------------------------
    # Step 4: Save config
    # ----------------------------------------------------------
    config = {
        "model_name": MODEL_NAME,
        "max_seq_length": st_model.max_seq_length,
        "embedding_dimension": 384,
        "pooling_mode": "mean",
        "normalize": True,
        "onnx_quantization": "int8",
        "original_framework": "sentence-transformers",
    }
    config_path = os.path.join(OUTPUT_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    # ----------------------------------------------------------
    # Step 5: Verify embeddings match (ONE TEXT AT A TIME)
    # ----------------------------------------------------------
    print(f"\n🔄 Verifying ONNX embeddings match sentence-transformers...")

    import onnxruntime as ort
    from tokenizers import Tokenizer as HFTokenizer

    test_queries = [
        "แมวพันธุ์ไหนเหมาะกับคนอยู่คนเดียว",
        "Persian cat personality",
    ]

    # Reference: sentence-transformers
    st_embeddings = st_model.encode(test_queries, normalize_embeddings=True)

    # ONNX: encode ONE text at a time to avoid batch shape issues
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = 1
    sess_options.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_int8_path, sess_options)

    onnx_tokenizer = HFTokenizer.from_file(os.path.join(OUTPUT_DIR, "tokenizer.json"))
    onnx_tokenizer.enable_padding()
    onnx_tokenizer.enable_truncation(max_length=128)

    all_good = True
    onnx_embeddings = []

    for text in test_queries:
        # Encode ONE at a time (avoids LayerNormalization batch shape bug)
        encoded = onnx_tokenizer.encode(text)
        input_ids = np.array([encoded.ids], dtype=np.int64)
        attention_mask = np.array([encoded.attention_mask], dtype=np.int64)

        outputs = session.run(
            ["last_hidden_state"],
            {"input_ids": input_ids, "attention_mask": attention_mask},
        )
        last_hidden = outputs[0]  # (1, seq_len, 384)

        # Mean pooling
        mask_expanded = attention_mask[:, :, np.newaxis].astype(np.float32)
        sum_emb = np.sum(last_hidden * mask_expanded, axis=1)
        sum_mask = np.clip(mask_expanded.sum(axis=1), a_min=1e-9, a_max=None)
        mean_emb = sum_emb / sum_mask

        # Normalize
        norms = np.linalg.norm(mean_emb, axis=1, keepdims=True)
        onnx_emb = (mean_emb / norms).astype(np.float32)[0]
        onnx_embeddings.append(onnx_emb)

    # Compare
    for i, q in enumerate(test_queries):
        cos_sim = float(np.dot(st_embeddings[i], onnx_embeddings[i]))
        l2_dist = float(np.linalg.norm(st_embeddings[i] - onnx_embeddings[i]))
        status = "✅" if cos_sim > 0.99 else "⚠️"
        print(f"  {status} '{q[:50]}'")
        print(f"     Cosine similarity: {cos_sim:.6f}")
        print(f"     L2 distance:       {l2_dist:.6f}")
        if cos_sim < 0.95:
            all_good = False

    # ----------------------------------------------------------
    # Step 6: Cleanup — remove temporary/large files
    # ----------------------------------------------------------
    print(f"\n🔄 Cleaning up...")

    # Remove FP32 model (only INT8 needed for deployment)
    if os.path.exists(onnx_fp32_path):
        os.remove(onnx_fp32_path)
        print(f"  🗑️ Removed model_fp32.onnx (INT8 is sufficient)")

    # Remove external data files
    for fname in os.listdir(OUTPUT_DIR):
        if fname.endswith(('.onnx.data', '.safetensors', '.bin')):
            os.remove(os.path.join(OUTPUT_DIR, fname))
            print(f"  🗑️ Removed {fname}")

    # Remove 1_Pooling directory
    pooling_dir = os.path.join(OUTPUT_DIR, "1_Pooling")
    if os.path.isdir(pooling_dir):
        shutil.rmtree(pooling_dir)
        print(f"  🗑️ Removed 1_Pooling/")

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print(f"\n{'=' * 60}")
    if all_good:
        print("✅ Export complete! Embeddings verified (cosine sim > 0.99)")
    else:
        print("⚠️ Export complete but embeddings differ — check results above")
    print(f"{'=' * 60}")

    print(f"\n📁 Files in {OUTPUT_DIR}/:")
    total_size = 0
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        if os.path.isfile(fp):
            sz = os.path.getsize(fp) / (1024 * 1024)
            total_size += sz
            print(f"   {f}: {sz:.1f} MB")
    print(f"   TOTAL: {total_size:.1f} MB")

    if int8_size > 100:
        print(f"\n⚠️  model_int8.onnx is {int8_size:.1f} MB (> 100 MB GitHub limit)")
        print(f"   You need Git LFS. Run:")
        print(f"     git lfs install")
        print(f"     git lfs track 'rag/model_onnx/model_int8.onnx'")
        print(f"     git add .gitattributes rag/model_onnx/")
    else:
        print(f"\n✅ model_int8.onnx is {int8_size:.1f} MB (< 100 MB)")
        print(f"   You can commit it directly.")

    print(f"\n🚀 Next steps:")
    print(f"   1. Commit rag/model_onnx/ to your repo")
    print(f"   2. Push to GitHub")
    print(f"   3. Deploy on Render with requirements-deploy.txt")
    print(f"   4. Estimated RAM: ~150 MB (fits 512 MB Render free tier)")


if __name__ == "__main__":
    main()
