---
license: cc-by-nc-4.0
base_model: meituan-longcat/LongCat-AudioDiT-1B
pipeline_tag: text-to-speech
library_name: transformers
datasets:
  - ghananlpcommunity/ghana-speech
language:
  - tw
  - ak
  - ee
  - ha
  - en
  - fat
  - dag
  - gjn
  - ada
  - any
  - avn
  - bud
  - bim
  - biv
  - bib
  - bwu
  - ncu
  - dga
  - mzw
  - ffm
  - acd
  - kbp
  - xsm
  - xon
  - kma
  - kus
  - lef
  - maw
  - naw
  - gur
  - nko
  - ntr
  - nzi
  - sig
  - sfw
  - lip
  - snw
  - sil
  - akp
  - tpm
  - kdh
  - bov
  - vag
tags:
  - text-to-speech
  - tts
  - audio
  - ghana
  - african-languages
  - voice-cloning
  - flow-matching
  - dit
  - africa-g2p
---

# Ghana AudioDiT

**Text-to-speech for 43 Ghanaian languages and Ghanaian English**, fine-tuned from
[LongCat-AudioDiT-1B](https://huggingface.co/meituan-longcat/LongCat-AudioDiT-1B).
It speaks with no reference audio, in a built-in speaker's voice, or in any voice you give it.

**Supported by** [Ghana NLP](https://ghananlp.org)

[Try it in the browser](https://huggingface.co/spaces/ghanaopenai/ghana-audiodit) ·
[Code, server and training scripts](https://github.com/GhanaOpenAI/ghana-audiodit)

## Listen

Held-out sentences (never seen in training) unless noted, in the two built-in modes.

| Language | Text | No voice prompt | Built-in speaker |
| --- | --- | --- | --- |
<!-- SAMPLES -->

## Quick start

```bash
pip install git+https://github.com/GhanaOpenAI/ghana-audiodit
```

```python
from ghana_audiodit import GhanaTTS

tts = GhanaTTS.from_pretrained("ghanaopenai/ghana-audiodit")   # ~6 GB download; ~4 GB VRAM

# 1. No voice prompt (default): the model picks a voice; reuse the seed to keep it
out = tts.synthesize("Akwaaba! Wo ho te sɛn ɛnnɛ?", language="Asante_Twi_twi")
out.save("twi.wav")
print(out.seed)

# 2. The language's built-in speaker
tts.synthesize("Woezɔ! Aleke nèfɔ ŋdi sia?", language="Ewe_ewe", mode="speaker").save("ewe.wav")

# 3. Your own voice: 3–15 s of one person speaking, plus what they said
tts.synthesize("Me ma wo akwaaba wɔ Cape Coast.", language="Fante_fat", mode="prompt",
               prompt_audio="my_voice.wav", prompt_text="Transcript of my_voice.wav",
               prompt_language="Fante_fat").save("fante.wav")
```

Write text in the **normal spelling** of the language. The package converts it to the
[africa-g2p](https://github.com/AfriSpeech/africa-g2p) *universal* spelling the model was
trained on, keeping English words as written, so code-switched text works too:

```python
tts.synthesize("Ɛnnɛ, Microsoft de Abilities for Jobs Program yi baeɛ.", language="Asante_Twi_twi")
```

`language` accepts a key (`"Asante_Twi_twi"`), a name (`"Ewe"`) or an ISO code (`"dag"`).
Long text is split into sentences; in no-prompt mode the first sentence's voice is reused for
the rest, so a passage keeps one voice. Options: `seed`, `steps` (16; more is slower and
slightly cleaner), `cfg_strength` (4.0), `speed` (1.0).

## Model size and hardware

| | |
| --- | --- |
| Parameters | **1.42 B**: diffusion transformer 982 M, UMT5-base text encoder 282 M, Wav-VAE 156 M |
| Download | 5.7 GB (fp32 weights) |
| GPU memory | **~4 GB** (default: transformer in bfloat16, as in training): 3.2 GB weights, 3.9 GB peak for an 18 s passage. `dtype="float32"`: ~6 GB. A 6 GB+ GPU is enough (T4, L4, A10G, RTX 3060 and up; on T4, which lacks bfloat16, use `dtype="float32"`) |
| Speed | H200, bfloat16, 16 steps: 18 s of speech in about 4 s; a short sentence in about 1–2 s |
| Output | 24 kHz mono |

bfloat16 and float32 score the same (universal CER 16.4 % vs 16.7 % on the held-out set below).
CPU inference works but is slow.

## Serving

This is a diffusion (flow-matching) model: it refines the whole utterance in 16 parallel denoising
steps rather than generating token by token, so LLM servers such as vLLM do not apply. The
[GitHub repository](https://github.com/GhanaOpenAI/ghana-audiodit) has a ready API server
(FastAPI; the one behind the demo) and a one-file [Modal](https://modal.com) deployment:

```bash
pip install "ghana-audiodit[server] @ git+https://github.com/GhanaOpenAI/ghana-audiodit"
uvicorn ghana_audiodit.server:app --host 0.0.0.0 --port 8210        # your own GPU machine

modal deploy deploy/modal_app.py                                     # or serverless on Modal (L4 GPU)
```

```bash
curl -X POST https://<your-endpoint>/synthesize -F "text=Akwaaba! Wo ho te sɛn?" \
     -F language=Asante_Twi_twi -F mode=noprompt -o out.wav
```

### Voice modes

| Mode | What it does | When to use it |
| --- | --- | --- |
| `noprompt` (default) | The voice comes from the random starting noise, set by `seed` | Variety; quick listening |
| `speaker` | Clones the language's built-in speaker (`speakers/`, one clear training-set voice per language) | A stable, native-sounding voice per language |
| `prompt` | Clones your reference recording | A specific voice |

### Without the package

The weights load with plain `transformers` (`trust_remote_code=True`), but you then have to
reproduce the text and audio preparation yourself: universal spelling via africa-g2p (with
`pyspellchecker` installed), the model's text normalisation, prompts at 24 kHz RMS-normalised
to −23 dBFS, and a duration in latent frames (`rates.json` gives frames per character per
language). The package does all of this; see `ghana_audiodit/tts.py`.

```python
from transformers import AutoModel, AutoTokenizer
model = AutoModel.from_pretrained("ghanaopenai/ghana-audiodit", trust_remote_code=True).cuda().eval()
tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)
```

## Languages

<!-- LANGUAGES -->

## Evaluation

Character error rate (CER) of [omniASR CTC-300M](https://github.com/facebookresearch/omnilingual-asr)
transcribing the model's speech, on held-out sentences in Asante Twi, Ewe and Dagbani (4 each),
cloned from a held-out speaker. Lower is better.

The model reads universal spelling, and omniASR writes what it hears in normal spelling. So both
the transcription and the reference are converted to universal spelling before comparing; this
ignores spelling-only differences such as *ɔ* vs *o*, which universal spelling merges anyway. The
CER on the normal-spelling text, without conversion, is shown for reference.

| | CER (universal spelling) | CER (normal spelling) |
| --- | --- | --- |
| Real recordings (omniASR's own error rate) | 19.8 % | 22.2 % |
| LongCat-AudioDiT-1B (base) | 31.9 % | 41.4 % |
| **This model** (step 20,000) | **16.2 %** | **23.2 %** |

The model's speech is as intelligible to omniASR as the real recordings. The set is small
(12 sentences), so treat differences of a couple of points as noise. Flow-matching validation
loss fell from 1.146 (base) to 0.915 with a prompt and from 1.208 to 0.940 without.

## Training

- **Data:** about 10 hours per language from
  [ghananlpcommunity/ghana-speech](https://huggingface.co/datasets/ghananlpcommunity/ghana-speech)
  (Hausa: 3 hours) — 199k clips, 407 hours, 43 languages. Most languages are read Bible
  text; English is conversational Ghanaian English.
- **Text:** transcripts converted to africa-g2p universal spelling (English left as is).
- **Audio:** resampled to 24 kHz and RMS-normalised to −23 dBFS. Many recordings were mastered
  loud (−13 to −15 dBFS), which made the base model's latents several times their normal size;
  normalising fixed it.
- **Method:** conditional flow matching with LoRA (rank 64, attention and feed-forward) plus
  full training of the small embedding, AdaLN and output layers (136M trainable parameters).
  85 % of examples had no voice prompt, so no-prompt generation is trained directly; text and
  prompt were dropped together 10 % of the time for classifier-free guidance, built the same
  way as inference builds its unconditional input.
- **Schedule:** AdamW (β 0.9/0.95), learning rate 1e-4 with 500 warm-up steps and cosine decay,
  batch 64, one H200. This checkpoint is step 20,000 (about 6.9 epochs), chosen for the lowest
  CER and validation loss.

## Fine-tuning

`lora/` holds the adapter and the fully trained layers of this checkpoint, so you can continue
training (for a new language, domain or voice) instead of starting from the base model. The
training pipeline (latent caching, universal-spelling manifests, the trainer, side CER
evaluation) is in [GhanaOpenAI/ghana-audiodit](https://github.com/GhanaOpenAI/ghana-audiodit)
under `training/`; see `training/README.md`.

## Files

| Path | Contents |
| --- | --- |
| `model.safetensors`, `config.json` | Merged model: base + this fine-tune (fp32) |
| `configuration_audiodit.py`, `modeling_audiodit.py` | Model code for `trust_remote_code` |
| `speakers/` | Built-in speaker per language (`<language>.wav`, `speakers.json`) |
| `rates.json` | Speaking rate per language (latent frames per character) |
| `lora/` | LoRA adapter + fully trained layers, for further fine-tuning |
| `samples/` | The audio on this page |

## Limitations

- Most training text is Bible readings, so the reading style leans formal.
- Write numbers as words: clips containing digits were excluded from training.
- In no-prompt mode the voice is random; a seed reproduces it for the same text, not across texts.
  For one stable voice, use `speaker` or `prompt` mode.
- English words inside Ghanaian text are pronounced from their English spelling, which the
  model saw mostly in the English portion of the data.
- Do not use this model to imitate a real person without their consent.

## License

The weights are released under **CC-BY-NC-4.0**, following the training data. The base model
LongCat-AudioDiT and its code are MIT-licensed (`LICENSE-longcat-audiodit`).

## Acknowledgements

Built by [Ghana Open AI](https://huggingface.co/ghanaopenai), supported by [Ghana NLP](https://ghananlp.org).
Thanks to Meituan for [LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT),
the [africa-g2p](https://github.com/AfriSpeech/africa-g2p) project for the universal spelling,
Meta for [Omnilingual ASR](https://github.com/facebookresearch/omnilingual-asr), and the
[zjubinchen](https://github.com/zjubinchen/LongCat-AudioDiT) fork whose LoRA trainer this
training code started from.
