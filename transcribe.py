#!/usr/bin/env python3
"""
Partie 2 : Transcription audio → texte (100% offline)
Usage : python transcribe.py [--turbo|--full] <fichier_audio> [fichier_sortie.txt]

  --turbo  (défaut) utilise ./models/whisper-large-v3-turbo
  --full            utilise ./models/whisper-large-v3

Formats supportés :
  .mp3                     — MP3 (torchcodec)
  .pcm / .raw / .l16       — PCM s16le mono 16 kHz brut
  .alaw / .al  / .pcma     — G.711 A-law  mono 8 kHz
  .ulaw / .ul  / .pcmu     — G.711 µ-law  mono 8 kHz

Exemples :
  python transcribe.py input.mp3
  python transcribe.py --full input.mp3 output.txt
"""
import argparse
import logging
import re
import sys

# Le pipeline Whisper ET generate() créent tous deux SuppressTokens* depuis
# generation_config (doublon interne à transformers, comportement correct).
# Le logger est émis depuis transformers.generation.utils via warning_once(),
# les filtres de logger parent ne se propagent pas — on cible le module exact.
_f = type("_NoSuppressTokensWarn", (logging.Filter,), {
    "filter": lambda self, r: "A custom logits processor" not in r.getMessage()
})()
logging.getLogger("transformers.generation.utils").addFilter(_f)

from pathlib import Path

import numpy as np
import torch
from torchcodec.decoders import AudioDecoder
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

MODEL_DIRS = {
    "turbo": Path("./models/whisper-large-v3-turbo"),
    "full":  Path("./models/whisper-large-v3"),
}
SAMPLE_RATE = 16_000  # Whisper attend du 16 kHz mono

# Extensions reconnues pour chaque format brut
_EXT_PCM16 = {".pcm", ".raw", ".l16"}
_EXT_PCMA  = {".alaw", ".al", ".pcma"}
_EXT_PCMU  = {".ulaw", ".ul", ".pcmu"}


def select_device() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float16
    return torch.device("cpu"), torch.float32


# ---------------------------------------------------------------------------
# Décodeurs G.711 (ITU-T)
# ---------------------------------------------------------------------------

def _decode_pcma(data: bytes) -> np.ndarray:
    """G.711 A-law → PCM int16."""
    raw  = (np.frombuffer(data, dtype=np.uint8).astype(np.int32)) ^ 0x55
    sign = (raw & 0x80) != 0
    exp  = (raw & 0x70) >> 4
    mant = raw & 0x0F
    # exp=0 : t = (mant*2+1)<<3  ;  exp>0 : t = (mant+16)<<(exp+3)
    t = np.where(exp == 0, (mant * 2 + 1) << 3, (mant + 16) << (exp + 3))
    return np.where(sign, t, -t).astype(np.int16)


def _decode_pcmu(data: bytes) -> np.ndarray:
    """G.711 µ-law → PCM int16."""
    raw       = (~np.frombuffer(data, dtype=np.uint8)).astype(np.int32) & 0xFF
    sign      = (raw & 0x80) != 0
    exp       = (raw & 0x70) >> 4
    mant      = raw & 0x0F
    magnitude = ((mant * 8) + 132) << exp   # (mant<<3 | 0x84) << exp
    linear    = magnitude - 132
    return np.where(sign, linear, -linear).astype(np.int16)


def _resample_8k_to_16k(samples: np.ndarray) -> np.ndarray:
    """Suréchantillonnage 8 kHz → 16 kHz par interpolation linéaire (torch)."""
    t = torch.from_numpy(samples).float().view(1, 1, -1)
    t = torch.nn.functional.interpolate(t, scale_factor=2.0, mode="linear", align_corners=False)
    return t.view(-1).numpy()


# ---------------------------------------------------------------------------
# Chargement audio unifié
# ---------------------------------------------------------------------------

def load_audio(path: str) -> dict:
    """Charge et normalise l'audio en float32 mono 16 kHz."""
    suffix = Path(path).suffix.lower()

    if suffix in _EXT_PCM16:
        # PCM s16le mono 16 kHz — lecture directe
        raw     = Path(path).read_bytes()
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0

    elif suffix in _EXT_PCMA:
        # G.711 A-law mono 8 kHz → décodage + suréchantillonnage
        raw     = Path(path).read_bytes()
        samples = _resample_8k_to_16k(_decode_pcma(raw).astype(np.float32) / 32768.0)

    elif suffix in _EXT_PCMU:
        # G.711 µ-law mono 8 kHz → décodage + suréchantillonnage
        raw     = Path(path).read_bytes()
        samples = _resample_8k_to_16k(_decode_pcmu(raw).astype(np.float32) / 32768.0)

    else:
        # Conteneurs (mp3, wav, flac, …) via torchcodec
        decoder = AudioDecoder(path, sample_rate=SAMPLE_RATE)
        audio   = decoder.get_all_samples().data  # Tensor (channels, samples)
        audio   = audio.mean(dim=0) if audio.shape[0] > 1 else audio.squeeze(0)
        samples = audio.numpy().astype(np.float32)

    return {"array": samples, "sampling_rate": SAMPLE_RATE}


def build_pipeline(device: torch.device, torch_dtype: torch.dtype, model_dir: Path):
    if not model_dir.exists():
        variant = "turbo" if "turbo" in model_dir.name else "full"
        flag = "" if variant == "turbo" else " --full"
        raise FileNotFoundError(
            f"Modèle introuvable dans '{model_dir}'.\n"
            f"Lancez d'abord : python download.py{flag}"
        )

    print(f"Chargement du modèle ({model_dir.name}) sur {device}...")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_dir,
        dtype=torch_dtype,
        low_cpu_mem_usage=True,
        local_files_only=True,  # aucun accès réseau
    )
    model.to(device)

    processor = AutoProcessor.from_pretrained(model_dir, local_files_only=True)

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        dtype=torch_dtype,
        device=device,
    )

    return model, processor, asr


# Tokens spéciaux Whisper à exclure des tokens de langue
_TASK_TOKENS = {
    "transcribe", "translate", "notimestamps", "nospeech",
    "startoftranscript", "endoftext", "startofprev", "startoflm",
}


def detect_language(model, processor, audio_array: np.ndarray, device, torch_dtype) -> str:
    """Détecte la langue principale depuis les 30 premières secondes.

    Utilise un forward pass avec le seul token SOT en entrée du décodeur
    pour éviter les interférences de forced_decoder_ids dans model.generate().
    """
    segment = audio_array[: SAMPLE_RATE * 30]
    input_features = processor(
        segment, sampling_rate=SAMPLE_RATE, return_tensors="pt"
    ).input_features.to(device, dtype=torch_dtype)

    decoder_input_ids = torch.tensor(
        [[model.config.decoder_start_token_id]], device=device
    )

    with torch.no_grad():
        logits = model(input_features, decoder_input_ids=decoder_input_ids).logits

    # logits[0, 0] = distribution sur le vocab pour le token suivant (position langue)
    next_token_logits = logits[0, 0]

    # Collecte des ids de tokens de langue (ex: <|fr|>, <|en|> — lettres minuscules seulement)
    vocab = processor.tokenizer.get_vocab()
    lang_ids: dict[int, str] = {}
    for token, token_id in vocab.items():
        m = re.match(r"^<\|([a-z]{2,8})\|>$", token)
        if m and m.group(1) not in _TASK_TOKENS:
            lang_ids[token_id] = m.group(1)

    if not lang_ids:
        return "unknown"

    ids_tensor = torch.tensor(list(lang_ids.keys()), device=device)
    best = next_token_logits[ids_tensor].argmax().item()
    return lang_ids[ids_tensor[best].item()]


def _fmt_time(seconds: float | None) -> str:
    """Formate les secondes en MM:SS.ss"""
    if seconds is None:
        return "??:??.??"
    m = int(seconds // 60)
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"


def _format_chunks(chunks: list) -> list[str]:
    lines = []
    prev_lang = None
    for chunk in chunks:
        ts = chunk.get("timestamp", (None, None))
        start = _fmt_time(ts[0] if ts else None)
        end   = _fmt_time(ts[1] if ts else None)
        lang  = chunk.get("language")

        # Affiche le tag de langue uniquement quand elle change
        lang_tag = f" [{lang}]" if lang and lang != prev_lang else ""
        if lang:
            prev_lang = lang

        lines.append(f"[{start} → {end}]{lang_tag}  {chunk['text'].strip()}")
    return lines


def transcribe(audio_path: str, model_dir: Path) -> dict:
    device, torch_dtype = select_device()
    model, processor, asr = build_pipeline(device, torch_dtype, model_dir)

    print(f"Chargement audio : {audio_path}")
    audio = load_audio(audio_path)
    audio_array = audio["array"]

    print("Détection de la langue principale...")
    global_language = detect_language(model, processor, audio_array, device, torch_dtype)
    print(f"Langue principale : {global_language}")

    print("Transcription en cours...")
    result = asr(
        audio,
        return_timestamps=True,
        generate_kwargs={"task": "transcribe"},  # pas de langue forcée → auto par fenêtre 30s
    )

    chunks = result.get("chunks", [])

    # Détection de langue par segment (1 forward pass encoder par chunk)
    print(f"Détection de langue par segment ({len(chunks)} segments)...")
    for chunk in chunks:
        ts = chunk.get("timestamp", (None, None))
        start_s = ts[0] if ts and ts[0] is not None else 0.0
        end_s   = ts[1] if ts and ts[1] is not None else len(audio_array) / SAMPLE_RATE
        seg = audio_array[int(start_s * SAMPLE_RATE) : int(end_s * SAMPLE_RATE)]
        # Segment < 1s : pas assez de signal, on garde la langue globale
        chunk["language"] = (
            detect_language(model, processor, seg, device, torch_dtype)
            if len(seg) >= SAMPLE_RATE
            else global_language
        )

    languages_used = sorted({c.get("language", global_language) for c in chunks})

    return {
        "text": result["text"].strip(),
        "language": global_language,
        "languages": languages_used,
        "chunks": chunks,
    }


def main():
    parser = argparse.ArgumentParser(description="Transcription audio offline (Whisper).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--turbo", action="store_true", help="Utilise whisper-large-v3-turbo (défaut)")
    group.add_argument("--full",  action="store_true", help="Utilise whisper-large-v3")
    parser.add_argument("audio",  help="Fichier audio à transcrire")
    parser.add_argument("output", nargs="?", help="Fichier texte de sortie (optionnel)")
    args = parser.parse_args()

    model_dir = MODEL_DIRS["full"] if args.full else MODEL_DIRS["turbo"]
    audio_path = args.audio
    output_path = args.output

    if not Path(audio_path).exists():
        print(f"Erreur : fichier introuvable : {audio_path}")
        sys.exit(1)

    result = transcribe(audio_path, model_dir)

    langs = result["languages"]
    print(f"\nLangue(s) : {', '.join(langs)}")
    print("\n--- Transcription ---")

    if result["chunks"]:
        for line in _format_chunks(result["chunks"]):
            print(line)
    else:
        print(result["text"])

    if output_path:
        lines = [f"Langue(s) : {', '.join(langs)}", ""]
        if result["chunks"]:
            lines.extend(_format_chunks(result["chunks"]))
        else:
            lines.append(result["text"])
        Path(output_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"\nSauvegardé dans : {output_path}")


if __name__ == "__main__":
    main()
