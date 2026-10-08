# Training

The pipeline that produced [ghanaopenai/ghana-audiodit](https://huggingface.co/ghanaopenai/ghana-audiodit):
LoRA fine-tuning of LongCat-AudioDiT-1B with conditional flow matching, on pre-encoded
Wav-VAE latents, on a single GPU.

```bash
pip install -e ".[train]"          # from the repository root
```

All scripts are run from the repository root (`python training/<script>.py`). The YAML files in
`configs/` use `/path/to/workdir`; edit the paths before running.

## 1. Data layout

One folder per language, each clip a `.wav` with a same-named `.lab` transcript in normal
spelling:

```
data/raw/Asante_Twi_twi/clip_0001.wav     data/raw/Asante_Twi_twi/clip_0001.lab
data/val_raw/Asante_Twi_twi/...           (a few held-out clips per language)
```

Folder names end in the africa-g2p language code (`<Name>_<code>`); `English_*` rows keep normal
spelling. Clips of 1–20 s; avoid digits in transcripts (write numbers as words).

## 2. Encode audio to latents (once)

```bash
python training/cache_latents.py --data_root data --out_dir work/latents
```

Resamples to 24 kHz, **RMS-normalises to −23 dBFS** (loud recordings otherwise give the VAE
latents several times their normal scale and the loss explodes), drops clipped audio, and encodes
each clip separately with the frozen Wav-VAE (padding a batch changes the last latent frames).
Writes `work/latents/{train,val}.jsonl`.

## 3. Universal-spelling transcripts

```bash
python training/make_universal_manifest.py --latent_dir work/latents   # → {train,val}_universal_enskip.jsonl
```

Converts transcripts to africa-g2p universal spelling, keeping English words (requires
`pyspellchecker`; `universal_text.py` refuses to run without it). The released model was
trained before the English skip was switched on, so its English words inside Ghanaian text
were converted too; new runs should use the `_enskip` manifests.

## 4. Train

```bash
python training/train_ghana.py --config training/configs/config_ghana_noprompt.yaml
```

The released settings (`config_ghana_noprompt.yaml`): LoRA rank 64 on attention and
feed-forward layers, full training of embeddings / AdaLN / output layers, batch 64,
learning rate 1e-4 (500 warm-up, cosine), 85 % of examples without a voice prompt
(`p_no_prompt`), text+prompt dropped together 10 % of the time for guidance (`p_uncond`).
Checkpoints, validation loss (with and without a prompt) and listening samples are written
every `save_every` steps; TensorBoard logs go to `<output_dir>/tb`. About 0.55 s/step on one
H200 for the 1B model.

**Continue from the released model** instead of the base: download it and point `init_from`
at its `lora/` folder (weights only; the optimizer and schedule start fresh):

```bash
huggingface-cli download ghanaopenai/ghana-audiodit --include "lora/*" --local-dir released
```
```yaml
training:
  init_from: released/lora
  learning_rate: 5.0e-5
```

## 5. Side evaluation (optional)

```bash
OMNIASR_SHERPA_DIR=/path/to/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12 \
python training/cer_watch.py --config training/configs/config_ghana_noprompt.yaml \
    --langs Asante_Twi_twi Ewe_ewe Dagbani_dag --per_lang 4 --draw 4 --name cer_3lang --every 4000
```

Runs next to training: for every new checkpoint it synthesises held-out sentences with and
without a prompt and scores them with omniASR CTC-300M (sherpa-onnx, CPU). Scores on the real
recordings are included as the floor. The prompted CER is the cleaner signal; no-prompt CER
also varies with which random voice a seed lands on.

## 6. Export

```bash
OMNIASR_SHERPA_DIR=... python training/select_speakers.py --manifest work/latents/train_universal.jsonl \
    --out release/speakers                 # one built-in speaker per language
python scripts/export_release.py --ckpt <run>/step_0020000 --train_manifest work/latents/train_universal.jsonl \
    --speakers release/speakers --out release
python scripts/make_card_samples.py --model release --val_manifest work/latents/val_universal.jsonl
python scripts/build_model_card.py --release release
```

`export_release.py` merges the LoRA into the base weights, checks the merge reproduces
base+LoRA, and adds `lora/`, `speakers/`, `rates.json` and the model code.

## Files

| File | Purpose |
| --- | --- |
| `cache_latents.py` | Audio → normalised 24 kHz → Wav-VAE latents |
| `make_universal_manifest.py`, `universal_text.py` | Transcripts → africa-g2p universal spelling |
| `dataset_ghana.py` | Latent dataset; random prompt/target split, optional no-prompt examples |
| `train_ghana.py` | Trainer (LoRA + full layers, flow matching, validation, samples) |
| `lora_utils.py` | LoRA injection / saving / loading (PEFT) |
| `sample.py`, `noprompt_test.py` | Listening samples with and without a prompt |
| `cer_watch.py` | Side CER evaluator (omniASR via sherpa-onnx) |
| `select_speakers.py` | Picks the clearest training clip per language as its built-in speaker |
