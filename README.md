# Ghana AudioDiT

Text-to-speech for **43 Ghanaian languages and Ghanaian English**, fine-tuned from
[LongCat-AudioDiT-1B](https://github.com/meituan-longcat/LongCat-AudioDiT). This repository has
the inference package, the API server, the web demo and the full training pipeline.

**Supported by** [Ghana NLP](https://ghananlp.org)

- Model: [ghanaopenai/ghana-audiodit](https://huggingface.co/ghanaopenai/ghana-audiodit) (with audio samples)
- Demo: [huggingface.co/spaces/ghanaopenai/ghana-audiodit](https://huggingface.co/spaces/ghanaopenai/ghana-audiodit)

## Languages

| Language | `language=` | Language | `language=` |
| --- | --- | --- | --- |
| Akuapem Twi | `Akuapem_Twi_twi` | Konni | `Konni_kma` |
| Anyin | `Anyin_any` | Kusaal | `Kusaal_kus` |
| Asante Twi | `Asante_Twi_twi` | Lelemi | `Lelemi_lef` |
| Avatime | `Avatime_avn` | Mampruli | `Mampruli_maw` |
| Bimoba | `Bimoba_bim` | Nawuri | `Nawuri_naw` |
| Bissa | `Bissa_bib` | Ninkare (Gurenɛ) | `Ninkare_gur` |
| Buli | `Buli_bwu` | Nkonya | `Nkonya_nko` |
| Chumburung | `Chumburung_ncu` | Ntcham (Bassar) | `Bassar_Ntcham_bud` |
| Dagaare | `Dagaare_dga` | Ntrubo | `Ntrubo_ntr` |
| Dagbani | `Dagbani_dag` | Nzema | `Nzema_nzi` |
| Dangme | `Dangme_ada` | Paasaal | `Paasaal_sig` |
| Deg | `Deg_mzw` | Sehwi | `Sehwi_sfw` |
| English (Ghanaian) | `English_eng` | Sekpele | `Sekpele_lip` |
| Ewe | `Ewe_ewe` | Selee | `Selee_snw` |
| Fante | `Fante_fat` | Sisaala (Tumulung) | `Sisaala_Tumulung_sil` |
| Fulfulde (Maasina) | `Fulfulde_Maasina_ffm` | Siwu | `Siwu_akp` |
| Gikyode | `Gikyode_acd` | Southern Birifor | `Birifor_Southern_biv` |
| Gonja | `Gonja_gjn` | Tampulma | `Tampulma_tpm` |
| Hausa | `Hausa_hau` | Tem | `Tem_kdh` |
| Kabiye | `Kabiye_kbp` | Tuwuli | `Tuwuli_bov` |
| Kasem | `Kasem_xsm` | Vagla | `Vagla_vag` |
| Konkomba | `Konkomba_xon` |  | |

`language` also accepts the display name (`"Ewe"`) or ISO 639-3 code (`"dag"`).

## Install

```bash
pip install git+https://github.com/GhanaOpenAI/ghana-audiodit
```

Python 3.10+, PyTorch 2.1+. The model has **1.42 B parameters** (diffusion transformer 982 M, UMT5-base
text encoder 282 M, Wav-VAE 156 M) and needs **~4 GB of GPU memory** (transformer in bfloat16, the
default; `dtype="float32"` needs ~6 GB, use it on T4s). Any 6 GB+ GPU works; CPU works but is slow.

## Use

```python
from ghana_audiodit import GhanaTTS

tts = GhanaTTS.from_pretrained()                       # ghanaopenai/ghana-audiodit

out = tts.synthesize("Akwaaba! Wo ho te sɛn ɛnnɛ?", language="Asante_Twi_twi")      # no prompt
out.save("twi.wav")

tts.synthesize("Woezɔ! Aleke nèfɔ ŋdi sia?", language="Ewe", mode="speaker").save("ewe.wav")

tts.synthesize("Me ma wo akwaaba wɔ Cape Coast.", language="Fante_fat", mode="prompt",
               prompt_audio="me.wav", prompt_text="What I say in me.wav").save("fante.wav")
```

| Mode | Voice |
| --- | --- |
| `noprompt` (default) | Chosen by the model from random noise; `seed` reproduces it |
| `speaker` | The language's built-in speaker, shipped with the model |
| `prompt` | Cloned from your 3–15 s reference recording (plus its transcript) |

Write in the normal spelling of the language, English words included. The package converts text
to the [africa-g2p](https://github.com/AfriSpeech/africa-g2p) universal spelling the model was
trained on (English words kept as written), splits long text into sentences, and estimates
durations from per-language speaking rates. See the language table above for `language` values.

## Repository

| Path | What |
| --- | --- |
| `ghana_audiodit/` | Inference package: `GhanaTTS`, text preparation, model code, API server (`server.py`) |
| `server/` | Supervisor and Cloudflare-tunnel scripts for running the API on your own machine |
| `deploy/` | One-file [Modal](https://modal.com) deployment of the API |
| `space/` | The static Hugging Face Space (one HTML page; API address in `config.json`) |
| `training/` | Latent caching, universal-spelling manifests, trainer, side CER evaluation, fine-tuning guide |
| `scripts/` | Release export, model-card samples and card building |

## Deploy as an API

The API (FastAPI) is the one behind the demo. This is a diffusion model, so it is served with plain
PyTorch, not an LLM server such as vLLM.

**Any GPU machine**

```bash
pip install "ghana-audiodit[server] @ git+https://github.com/GhanaOpenAI/ghana-audiodit"
OMNIASR_SHERPA_REPO=csukuangfj/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12 \
uvicorn ghana_audiodit.server:app --host 0.0.0.0 --port 8210
```

`OMNIASR_SHERPA_REPO` is optional: with it, reference audio uploaded without a transcript is
transcribed automatically. For a long-running service, `server/systemd/` has example units that start the API and a
Cloudflare tunnel on boot and restart them on failure; `server/run_supervised.sh` and
`server/run_tunnel.sh` do the same without systemd.

**Modal (serverless GPU)**

```bash
pip install modal && modal setup
modal deploy deploy/modal_app.py      # → https://<workspace>--ghana-audiodit-api.modal.run
```

Runs on an L4, scales to zero when idle, and caches the model in a Modal Volume.

**Endpoints**

| Method | Path | |
| --- | --- | --- |
| GET | `/health` | Status, queue length |
| GET | `/languages` | The 43 languages and their built-in speakers |
| GET | `/speakers/{language}.wav` | Built-in speaker preview |
| POST | `/synthesize` | Multipart form: `text`, `language`, `mode` (`noprompt` / `speaker` / `prompt`), `prompt_audio`, `prompt_text`, `seed`, `steps`, `cfg_strength`, `speed` → WAV |

```bash
curl -X POST https://<your-endpoint>/synthesize -F "text=Akwaaba! Wo ho te sɛn?" \
     -F language=Asante_Twi_twi -o out.wav
```

To use the web demo with your endpoint, set its address in `space/config.json`.

## Train

See [training/README.md](training/README.md). The released model: about 10 hours per language
(407 hours) from [ghananlpcommunity/ghana-speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech),
LoRA rank 64 plus full training of the small layers, 85 % of examples without a voice prompt,
20,000 steps at batch 64 on one H200.

## License

Code: MIT (`LICENSE`); `ghana_audiodit/audiodit/` is from LongCat-AudioDiT, MIT
(`LICENSE-longcat-audiodit`). Model weights: CC-BY-NC-4.0, following the training data.

Built by [Ghana Open AI](https://huggingface.co/ghanaopenai), supported by [Ghana NLP](https://ghananlp.org).
