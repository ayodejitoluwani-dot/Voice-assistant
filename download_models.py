"""
Download models to local disk so the app can run fully offline afterwards
(no repeated Hugging Face calls, no waiting on network every time you
start the app).

Requirements:
  - You must already have Hugging Face access granted for each gated
    NCAIR model (visit each model's page on huggingface.co and request
    access first -- this script cannot skip that step).
  - HUGGINGFACE_TOKEN set in your environment (a token with read access
    is enough): export HUGGINGFACE_TOKEN=hf_xxxxxxxx

Usage:
  python download_models.py            # downloads all 3 ASR languages
  python download_models.py hausa      # downloads just one ASR model
  python download_models.py ncair-llm  # downloads the N-ATLaS agent LLM (GGUF, ~4.6GB)

ASR models are saved under ./models/<language>/. asr.py automatically
checks this folder first and only falls back to Hugging Face if it's
missing. The LLM is saved under ./models/ncair-llm/ -- agent_ncair.py
checks there before trying to load anything.
"""

import os
import sys

from huggingface_hub import snapshot_download, hf_hub_download

NCAIR_ASR_MODEL_MAP = {
    "hausa": "NCAIR1/Hausa-ASR",
    "yoruba": "NCAIR1/Yoruba-ASR",
    "igbo": "NCAIR1/Igbo-ASR",  # unconfirmed model id -- verify on Hugging Face
}

# Quantized (GGUF) version of NCAIR1/N-ATLaS, ~4.6GB -- far more practical
# for CPU-only laptops than the ~20GB full-precision weights. This is a
# community requantization; if it's ever taken down or fails, the full
# model is at NCAIR1/N-ATLaS (needs Hugging Face access request first,
# same as the ASR models).
NCAIR_LLM_REPO = "inuwamobarak/N-ATLaS-8B-GGUF-Q4_K_M"
NCAIR_LLM_FILENAME = "ggml-model-Q4_K_M.gguf"

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def download_asr(lang: str):
    repo_id = NCAIR_ASR_MODEL_MAP[lang]
    dest = os.path.join(MODELS_DIR, lang)
    token = os.environ.get("HUGGINGFACE_TOKEN") or None

    if not token:
        print("WARNING: HUGGINGFACE_TOKEN is not set. This will fail for gated models.")
        print("Set it first: export HUGGINGFACE_TOKEN=hf_xxxxxxxx")

    print(f"Downloading {repo_id} -> {dest} ...")
    try:
        snapshot_download(repo_id=repo_id, local_dir=dest, token=token)
        print(f"Done: {lang}")
    except Exception as e:
        print(f"FAILED for {lang}: {e}")
        print("If this says 403 / gated repo, you don't have access approved yet --")
        print(f"visit https://huggingface.co/{repo_id} and request access.")


def download_ncair_llm():
    dest_dir = os.path.join(MODELS_DIR, "ncair-llm")
    os.makedirs(dest_dir, exist_ok=True)
    token = os.environ.get("HUGGINGFACE_TOKEN") or None

    print(f"Downloading {NCAIR_LLM_REPO}/{NCAIR_LLM_FILENAME} -> {dest_dir} ...")
    print("This is ~4.6GB -- it will take a while, similar to the ASR model downloads.")
    try:
        path = hf_hub_download(
            repo_id=NCAIR_LLM_REPO,
            filename=NCAIR_LLM_FILENAME,
            local_dir=dest_dir,
        )
        # agent_ncair.py expects this exact filename -- rename if the hub
        # gave it something else (e.g. nested in a subfolder).
        expected = os.path.join(dest_dir, "n-atlas-q4_k_m.gguf")
        if os.path.abspath(path) != os.path.abspath(expected):
            os.replace(path, expected)
        print(f"Done: ncair-llm -> {expected}")
    except Exception as e:
        print(f"FAILED for ncair-llm: {e}")
        print(
            "If this fails with a gated/403 error, first request access to the base "
            "model at https://huggingface.co/NCAIR1/N-ATLaS -- some quantized "
            "mirrors require that first."
        )


if __name__ == "__main__":
    targets = sys.argv[1:] or list(NCAIR_ASR_MODEL_MAP)

    for target in targets:
        if target == "ncair-llm":
            download_ncair_llm()
        elif target in NCAIR_ASR_MODEL_MAP:
            download_asr(target)
        else:
            print(f"Unknown target '{target}'. Choose from: {list(NCAIR_ASR_MODEL_MAP) + ['ncair-llm']}")
