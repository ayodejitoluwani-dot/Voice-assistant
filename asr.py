"""
ASR wrapper for NCAIR's Yoruba, Hausa, and Igbo models.

Returns a confidence score (0-1) alongside the transcript, derived from the
model's own token probabilities during generation -- there's no separate
confidence model, this is the model's own certainty about what it produced.

If the NCAIR models are gated on Hugging Face, set HUGGINGFACE_TOKEN in the
environment (the Colab notebook's Step 3 does this for you) and accept the
model's usage terms on its Hugging Face page first.
"""

import io
import math
import os

NCAIR_MODEL_MAP = {
    "hausa": "NCAIR1/Hausa-ASR",
    "yoruba": "NCAIR1/Yoruba-ASR",
    "igbo": "NCAIR1/Igbo-ASR",  # unconfirmed model id -- verify on Hugging Face
}

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

_model_cache = {}


def _get_model(lang: str):
    if lang not in NCAIR_MODEL_MAP:
        raise ValueError(f"Unsupported language '{lang}'. Available: {list(NCAIR_MODEL_MAP)}")

    if lang not in _model_cache:
        from transformers import WhisperProcessor, WhisperForConditionalGeneration

        local_path = os.path.join(MODELS_DIR, lang)
        if os.path.isdir(local_path) and os.listdir(local_path):
            # Downloaded via download_models.py -- load fully offline, no HF call.
            source = local_path
            load_kwargs = {"local_files_only": True}
        else:
            # Not downloaded yet -- fetch from the Hugging Face Hub (needs
            # access granted + HUGGINGFACE_TOKEN for gated models).
            source = NCAIR_MODEL_MAP[lang]
            load_kwargs = {"token": os.environ.get("HUGGINGFACE_TOKEN") or None}

        processor = WhisperProcessor.from_pretrained(source, **load_kwargs)
        model = WhisperForConditionalGeneration.from_pretrained(source, **load_kwargs)
        model.eval()
        _model_cache[lang] = (processor, model)

    return _model_cache[lang]


def _load_audio(audio_path: str):
    """Load an audio file and resample to 16kHz. Uses pydub (ffmpeg) first
    so compressed/browser formats work, falling back to librosa directly."""
    import librosa

    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(audio_path)
        buf = io.BytesIO()
        segment.export(buf, format="wav")
        buf.seek(0)
        audio_np, sr = librosa.load(buf, sr=16000)
    except Exception:
        audio_np, sr = librosa.load(audio_path, sr=16000)

    return audio_np, sr


def transcribe(lang: str, audio_path: str) -> dict:
    """
    Transcribe an audio file using the NCAIR model for the given language.

    Returns:
        {"text": str, "confidence": float} -- confidence is in [0, 1],
        derived from exp(average log-probability) of the generated tokens.
    """
    import torch

    audio_np, sr = _load_audio(audio_path)
    processor, model = _get_model(lang)
    inputs = processor(audio_np, sampling_rate=sr, return_tensors="pt")

    with torch.no_grad():
        output = model.generate(
            inputs.input_features,
            attention_mask=inputs.get("attention_mask"),
            max_new_tokens=225,
            output_scores=True,
            return_dict_in_generate=True,
        )

    text = processor.batch_decode(output.sequences, skip_special_tokens=True)[0].strip()

    generated_ids = output.sequences[0][1:]
    step_logprobs = []
    for step_logits, token_id in zip(output.scores, generated_ids):
        log_probs = torch.log_softmax(step_logits[0], dim=-1)
        step_logprobs.append(log_probs[token_id].item())

    avg_logprob = sum(step_logprobs) / len(step_logprobs) if step_logprobs else float("-inf")
    confidence = math.exp(avg_logprob) if avg_logprob != float("-inf") else 0.0

    return {"text": text, "confidence": confidence}
