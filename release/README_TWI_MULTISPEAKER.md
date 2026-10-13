---
language:
- tw
- ak
license: cc-by-nc-4.0
tags:
- text-to-speech
- tts
- speech-synthesis
- voice-cloning
- zero-shot
- twi
- akan
- ghana
- audiodit
- longcat
pipeline_tag: text-to-speech
---

# Twi Multi-Speaker AudioDiT 🇬🇭🎙️

**Author:** [Ghana Open AI](https://huggingface.co/ghanaopenai)  
**Supported by:** [Ghana NLP](https://ghananlp.org)  

A multi-speaker, voice-prompted diffusion text-to-speech model for **Twi (Akan)** with **zero-shot voice cloning**, based on **LongCat-AudioDiT-1B** and fine-tuned on the high-fidelity [ghanaopenai/twi-multispeaker-tts](https://huggingface.co/datasets/ghanaopenai/twi-multispeaker-tts) corpus.

Unlike text-only TTS models, this model is **100% voice-prompted**: it conditions on a reference voice audio clip (3–6 seconds) to synthesize new text in that exact speaker's voice, pitch, and timbre.

---

## 🌟 Key Highlights

- **Architecture:** LongCat-AudioDiT-1B with Conditional Flow Matching (CFM) on Wav-VAE latents.
- **Dataset:** Fine-tuned on **32.5 hours** of pure vocal audio across **3,895 single-speaker clips** filtered to **UTMOS >= 2.2** (studio broadcast tier).
- **Spelling:** Trained on `africa-g2p` universal spelling with English loanword retention.
- **Evaluation:** Evaluated with `omniASR CTC-300M` achieving **13.92% Universal Character Error Rate (CER)**, within 1.3% of the real human speech error floor (12.58%).
- **Packaged Presets:** Includes 6 curated reference voices in `voices/` (`bright_low`, `deep_excited`, `deep_calm`, `warm_storyteller`, `lively_host`, `bright_clear`).

---

## 🎙️ Included Preset Voices

| Voice ID | F0 (Pitch) | Timbre Profile |
|---|---|---|
| `deep_calm` | 96.9 Hz | Deep, authoritative baritone voice with calm cadence |
| `deep_excited` | 131.6 Hz | Rich, dynamic baritone with energetic delivery |
| `warm_storyteller` | 169.6 Hz | Warm narrative timbre with clear, expressive pacing |
| `bright_low` | 211.3 Hz | Rich, grounded alto voice with natural resonance |
| `lively_host` | 226.1 Hz | Upbeat, energetic broadcast host with vibrant cadence |
| `bright_clear` | 254.2 Hz | Crisp, bright feminine tone with articulate clarity |

---

## 💻 Usage

```python
import torch
from ghana_audiodit.audiodit import AudioDiTModel
from transformers import AutoTokenizer

model = AudioDiTModel.from_pretrained("ghanaopenai/twi-multispeaker-audiodit", trust_remote_code=True).cuda()
model.vae.to_half()
tokenizer = AutoTokenizer.from_pretrained(model.config.text_encoder_model)

# Synthesize with a reference prompt audio
# prompt_audio: (1, 1, num_samples) at 24000 Hz
# text: prompt_text + " " + target_text
inputs = tokenizer(["Prompt text here. Target text to speak."], return_tensors="pt")
out = model(
    input_ids=inputs.input_ids.cuda(),
    attention_mask=inputs.attention_mask.cuda(),
    prompt_audio=prompt_audio.cuda(),
    steps=16,
    cfg_strength=4.0,
    guidance_method="apg"
)
audio = out.waveform.squeeze().cpu().numpy()
```

---

## 📜 Citation & Credits

- **Base Architecture:** [Meituan LongCat-AudioDiT](https://github.com/meituan-longcat/LongCat-AudioDiT)
- **Dataset:** [ghanaopenai/twi-multispeaker-tts](https://huggingface.co/datasets/ghanaopenai/twi-multispeaker-tts)
