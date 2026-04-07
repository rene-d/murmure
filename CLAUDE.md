# CLAUDE.md

Offline audio transcription. Whisper large-v3-turbo. Download once, transcribe forever (no net).

## Stack

Python ≥3.14 · uv · transformers · torch · torchcodec · model in `./models/whisper-large-v3-turbo`

## Files

- `download.py` — fetch model from HuggingFace, save local (needs internet)
- `transcribe.py` — transcribe audio, fully offline (`local_files_only=True`)
- `w.py` — scratch/prototype, online, not production
- `.env` — `HF_TOKEN` (optional, gated models only)

## Commands

```bash
uv sync
uv run python download.py            # step 1 — internet required
uv run python transcribe.py input.mp3
uv run python transcribe.py input.mp3 output.txt
```

## Notes

- Run `download.py` before `transcribe.py` — model dir must exist
- Device: CUDA → MPS → CPU, float16 on GPU/MPS, float32 on CPU
- `suppress_tokens` cleared in generation_config — avoids duplicate logits processor warnings
