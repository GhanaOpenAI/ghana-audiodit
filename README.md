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

## Models

1. **[ghanaopenai/ghana-audiodit](https://huggingface.co/ghanaopenai/ghana-audiodit)** — Text-to-speech for 43 Ghanaian languages. Speaks with random seed-based voices.
2. **[ghanaopenai/twi-multispeaker-audiodit](https://huggingface.co/ghanaopenai/twi-multispeaker-audiodit)** — Multi-speaker Twi TTS with **zero-shot voice cloning** and packaged preset voices (`bright_low`, `deep_excited`, `deep_calm`, `warm_storyteller`, `lively_host`, `bright_clear`), fine-tuned on [ghanaopenai/twi-multispeaker-tts](https://huggingface.co/datasets/ghanaopenai/twi-multispeaker-tts).

## Use

### 1. Multi-Speaker Twi with Voice Cloning & Presets

```python
from ghana_audiodit import GhanaTTS

# Load the Twi multi-speaker prompted model
tts = GhanaTTS.from_pretrained("ghanaopenai/twi-multispeaker-audiodit")

# Synthesize using a packaged preset voice ('bright_low', 'deep_excited', 'deep_calm', etc.)
out = tts.synthesize("Akwaaba! Wo ho te sɛn ɛnnɛ?", voice="bright_low")
out.save("twi_bright.wav")

# Or zero-shot clone any reference voice audio (3–6 seconds):
import torch
from ghana_audiodit.utils import load_audio
ref_wav = load_audio("path/to/my_voice.wav", sr=24000)

out_clone = tts.synthesize(
    "Dɔketa no maa ɔyarefoɔ no aduro wɔ asopiti hɔ.",
    custom_prompt_audio=ref_wav,
    custom_prompt_text="Prompt speaker speech transcript here."  # optional, auto-transcribes if omitted
)
out_clone.save("cloned_twi.wav")
```

### 2. 43-Language Base Model

```python
from ghana_audiodit import GhanaTTS

tts = GhanaTTS.from_pretrained("ghanaopenai/ghana-audiodit")

out = tts.synthesize("Akwaaba! Wo ho te sɛn ɛnnɛ?", language="Asante_Twi_twi", seed=42)
out.save("twi.wav")

tts.synthesize("Woezɔ! Aleke nèfɔ ŋdi sia?", language="Ewe", seed=42).save("ewe.wav")
```

Write in the normal spelling of the language, English words included. The package converts text
to the [africa-g2p](https://github.com/AfriSpeech/africa-g2p) universal spelling the model was
trained on (English words kept as written), splits long text into sentences, and estimates
durations from per-language speaking rates. See the language table above for `language` values.

## Repository

| Path | What |
| --- | --- |
| `ghana_audiodit/` | Inference package: `GhanaTTS`, text preparation, model code, API server (`server.py`) |
| `server/` | Supervisor and Cloudflare-tunnel scripts for running the API on your own machine |
| `Dockerfile`, `docker/` | Self-contained GPU image of the API (model included) |
| `deploy/` | Optional [Modal](https://modal.com) example |
| `space/` | The static Hugging Face Space (one HTML page; API address in `config.json`) |
| `training/` | Latent caching, universal-spelling manifests, trainer, side CER evaluation, fine-tuning guide |
| `scripts/` | Release export, model-card samples and card building |

## Deploy as an API

The API (FastAPI) is the one behind the demo. This is a diffusion model, so it is served with plain
PyTorch, not an LLM server such as vLLM.

**Docker (recommended)** — the image contains the model and the exact tested dependency versions,
so it runs offline on any machine with an NVIDIA GPU (≥ 6 GB) and the NVIDIA container toolkit
(10.1 GB download):

```bash
docker run --gpus all -p 8000:8000 ghcr.io/ghanaopenai/ghana-audiodit:latest
curl -X POST localhost:8000/synthesize -F "text=Akwaaba! Wo ho te sɛn?" -F language=Asante_Twi_twi -o out.wav
```

`ghcr.io/ghanaopenai/ghana-audiodit:slim` (4.6 GB download) leaves the model out and downloads it
on first start (mount a volume at `/models` to keep it). Build either yourself with `docker build .`
(`--build-arg BAKE_MODEL=0` for slim). The same image works on Modal, RunPod, Kubernetes and the like;
`deploy/modal_app.py` is an optional Modal example.

**Without Docker**

```bash
pip install "ghana-audiodit[server] @ git+https://github.com/GhanaOpenAI/ghana-audiodit"
uvicorn ghana_audiodit.server:app --host 0.0.0.0 --port 8000
```

For a long-running service, `server/systemd/` has example units that start the API and a
Cloudflare tunnel on boot and restart them on failure; `server/run_supervised.sh` and
`server/run_tunnel.sh` do the same without systemd.

**Endpoints**

| Method | Path | |
| --- | --- | --- |
| GET | `/health` | Status, queue length |
| GET | `/languages` | The 43 languages |
| POST | `/synthesize` | Form fields: `text`, `language`, `seed`, `steps`, `cfg_strength`, `speed` → WAV |

To use the web demo with your endpoint, set its address in `space/config.json`.

## Train

See [training/README.md](training/README.md). The released model: about 10 hours per language
(407 hours) from [ghananlpcommunity/ghana-speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech),
LoRA rank 64 plus full training of the small layers, 85 % of examples without a voice prompt,
batch 64 on one H200; the released checkpoint is step 28,000 of a 40,000-step run.

## License

Code: MIT (`LICENSE`); `ghana_audiodit/audiodit/` is from LongCat-AudioDiT, MIT
(`LICENSE-longcat-audiodit`). **Model weights, including the Docker images that contain them:
CC-BY-NC-4.0** (`LICENSE-MODEL`), following the training data.

Built by [Ghana Open AI](https://huggingface.co/ghanaopenai), supported by [Ghana NLP](https://ghananlp.org).
