# Transcription audio offline

Transcription locale via **Whisper large-v3-turbo**. Télécharger une fois, transcrire sans réseau.

## Prérequis

- Python ≥ 3.14
- [uv](https://github.com/astral-sh/uv)
- GPU optionnel (CUDA ou MPS) — fonctionne aussi sur CPU

## Installation

```bash
uv sync
```

## Utilisation

### Étape 1 — Télécharger le modèle (internet requis, une seule fois)

```bash
uv run python download.py
```

Sauvegarde le modèle dans `./models/whisper-large-v3-turbo`. Après ça, plus besoin de réseau.

### Étape 2 — Transcrire

```bash
uv run python transcribe.py audio.mp3
uv run python transcribe.py audio.mp3 sortie.txt
```

## Formats supportés

| Extension | Format |
|-----------|--------|
| `.mp3`, `.wav`, `.flac`, … | Conteneurs audio (via torchcodec) |
| `.pcm`, `.raw`, `.l16` | PCM s16le mono 16 kHz brut |
| `.alaw`, `.al`, `.pcma` | G.711 A-law mono 8 kHz |
| `.ulaw`, `.ul`, `.pcmu` | G.711 µ-law mono 8 kHz |

## Sortie

Segments horodatés avec détection de langue par segment :

```text
[00:00.00 → 00:04.32] [fr]  Bonjour, voici la transcription.
[00:04.32 → 00:09.10] [en]  And here it switches to English.
```

## Accélération matérielle

Détection automatique : CUDA → MPS → CPU. Float16 sur GPU/MPS, float32 sur CPU.

## HuggingFace Token

Optionnel (modèles gatés uniquement). Créer un `.env` :

```shell
HF_TOKEN=hf_...
```

## Samples

Les samples sont extraits de [Test sequences for EVS codec - TS 26.444](https://www.3gpp.org/ftp/Specs/archive/26_series/26.444/test_sequences/26444-c00-TestSeq.zip).

```shell
unzip -oq 26444-c00-TestSeq.zip 'testvec/testv/input/amrwb/T*.INP' testvec/testv/input/wb/stv16c.INP

mkdir -p samples/{pcm,mp3,pcma,pcmu}

for input in testvec/testv/input/amrwb/T*.INP testvec/testv/input/wb/stv16c.INP ; do

    name=$(basename $input .INP)

    cp -f $input samples/pcm/$name.pcm

    # Conversion PCM linéaire 16 bits signés à 16 kHz en .mp3
    ffmpeg -y -v +error -f s16le -ar 16000 -ac 1 -i $input samples/mp3/$name.mp3

    # Conversion PCM linéaire 16 bits signés à 16 kHz en G.711 A-law (PCMA)
    ffmpeg -y -v +error -f s16le -ar 16000 -ac 1 -i $input \
         -ar 8000 -c:a pcm_alaw -f alaw samples/pcma/$name.pcma

    # Conversion PCM linéaire 16 bits signés à 16 kHz en G.711 µ-law (PCMU)
    ffmpeg -y -v +error -f s16le -ar 16000 -ac 1 -i $input \
         -ar 8000 -c:a pcm_mulaw -f mulaw samples/pcmu/$name.pcmu
done
```
