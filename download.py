#!/usr/bin/env python3
"""
Partie 1 : Téléchargement du modèle Whisper (nécessite une connexion internet)
Usage : python download.py [--full]
  --full   Télécharge openai/whisper-large-v3 (plus grand, plus précis)
           Par défaut : openai/whisper-large-v3-turbo (plus rapide)
"""
import argparse
import os
from pathlib import Path

import torch
from dotenv import load_dotenv
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

load_dotenv()

MODELS = {
    "turbo": ("openai/whisper-large-v3-turbo", Path("./models/whisper-large-v3-turbo")),
    "full":  ("openai/whisper-large-v3",       Path("./models/whisper-large-v3")),
}


def main():
    parser = argparse.ArgumentParser(description="Télécharge un modèle Whisper localement.")
    parser.add_argument("--full", action="store_true", help="Télécharge whisper-large-v3 au lieu de whisper-large-v3-turbo")
    args = parser.parse_args()

    model_id, model_dir = MODELS["full"] if args.full else MODELS["turbo"]

    token = os.getenv("HF_TOKEN") or None
    if token:
        print("HF_TOKEN détecté.")

    print(f"Téléchargement de {model_id}...")
    print(f"Destination : {model_dir.resolve()}\n")

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id,
        dtype=torch.float16,
        low_cpu_mem_usage=True,
        use_safetensors=True,
        token=token,
    )

    processor = AutoProcessor.from_pretrained(model_id, token=token)

    model_dir.mkdir(parents=True, exist_ok=True)

    print("\nSauvegarde du modèle...")
    model.save_pretrained(model_dir)

    print("Sauvegarde du processeur (tokenizer + feature extractor)...")
    processor.save_pretrained(model_dir)

    print(f"\nModèle prêt dans : {model_dir.resolve()}")
    print("Vous pouvez maintenant utiliser transcribe.py sans connexion internet.")


if __name__ == "__main__":
    main()
