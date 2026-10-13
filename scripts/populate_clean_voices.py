import os
import json
import soundfile as sf
import torch
from ghana_audiodit.utils import normalize_loudness, load_audio

out_dir = "/mnt/volume_d2wey28/projects/longcat-audiodit-ghana/release/twi-multispeaker-audiodit/voices"
os.makedirs(out_dir, exist_ok=True)
sr = 24000

voices = [
    {
        "id": "bright_low",
        "display_name": "Bright Low",
        "description": "Rich, grounded alto voice with natural resonance",
        "pitch_hz": 213.4,
        "file": "bright_low.wav",
        "source_wav": "/mnt/volume_d2wey28/projects/fish-s2-ghana/data10r/val_raw/Asante_Twi_twi/Asante_Twi_twi_00104506.wav",
        "prompt_text": "Na ɛbaa sɛ nna a wɔrebɛfa no akɔ soro reyɛ awie duru no, ɔde sii n’ani sɛ ɔrekɔ Yerusalem.",
        "prompt_text_universal": "na ebaa se nna a worebefa no ako osoro reye awie duru no, ode sii nani se oreko yerusalem."
    },
    {
        "id": "deep_excited",
        "display_name": "Deep Excited",
        "description": "Rich, dynamic baritone with energetic delivery",
        "pitch_hz": 114.4,
        "file": "deep_excited.wav",
        "source_wav": "/mnt/volume_d2wey28/projects/longcat-audiodit-ghana/checkpoints/lora_r64_twi_prompted/samples/prompts/Asante_Twi_twi_prompt.wav",
        "prompt_text": "No, ɛkyerɛ no sɛ prostate no, sɛdeɛ Nyame ahyɛhyɛ no no, ɔse sɛ cells no multiply, na ɛmultiply dubaabi a, ɔse sɛ ɔmo gyae na ebi wu na foforɔ bɛka ho.",
        "prompt_text_universal": "no, echere no se prostate no, sedie nyame asheshe no no, ose se cells no multiply, na emultiply dubaabi a, ose se omo jae na ebi wu na foforo beka ho."
    },
    {
        "id": "deep_calm",
        "display_name": "Deep Calm",
        "description": "Deep, authoritative baritone voice with calm cadence",
        "pitch_hz": 115.0,
        "file": "deep_calm.wav",
        "source_wav": "/mnt/volume_d2wey28/projects/fish-s2-ghana/data10r/val_raw/Asante_Twi_twi/Asante_Twi_twi_00046692.wav",
        "prompt_text": "Na wowie yeinom a, sane da wo mfe nifa so soa Yuda fie amumuyɛ adaduanan; mede afe baako biara mayɛ wo da koro.",
        "prompt_text_universal": "na wowie yeinom a, sane da wo mfe nifa so soa yuda fie amumuye adaduanan; mede afe baako biara maye wo da koro."
    },
    {
        "id": "warm_storyteller",
        "display_name": "Warm Storyteller",
        "description": "Warm narrative timbre with clear, expressive pacing",
        "pitch_hz": 173.7,
        "file": "warm_storyteller.wav",
        "source_wav": "/mnt/volume_d2wey28/projects/fish-s2-ghana/data10r/val_raw/Asante_Twi_twi/Asante_Twi_twi_00063401.wav",
        "prompt_text": "Mfuturo mu na w’asɛm bɛfiri ayɛ huhuhuhu; wo nne bɛyɛ sɛ ɔsaman deɛ afiri fam.",
        "prompt_text_universal": "mfuturo mu na wasem befiri aye huhuhuhu; wo nne beye se osaman die afi fam."
    },
    {
        "id": "lively_host",
        "display_name": "Lively Host",
        "description": "Upbeat, energetic broadcast host with vibrant cadence",
        "pitch_hz": 204.7,
        "file": "lively_host.wav",
        "source_wav": "/mnt/volume_d2wey28/projects/fish-s2-ghana/data10r/val_raw/Asante_Twi_twi/Asante_Twi_twi_00116870.wav",
        "prompt_text": "Na afei, hwɛ, merekɔ me man mu: bra na menkasa menkyerɛ wo deɛ ɔman yi bɛyɛ wo man nna a ɛdi akyire no mu.",
        "prompt_text_universal": "na afei, hwe, mereko me man mu: bra na menkasa menchere wo die oman yi beye wo man nna a edi akyire no mu."
    },
    {
        "id": "bright_clear",
        "display_name": "Bright Clear",
        "description": "Crisp, bright feminine tone with articulate clarity",
        "pitch_hz": 193.9,
        "file": "bright_clear.wav",
        "source_wav": "/mnt/volume_d2wey28/projects/fish-s2-ghana/data10r/val_raw/Asante_Twi_twi/Asante_Twi_twi_00115918.wav",
        "prompt_text": "Mede mo asɔfodie no mama mo sɛ akyɛdeɛ som adwuma; na obi foforɔ biara a ɔbɛbɛpɛn ho no, wɔnku no.",
        "prompt_text_universal": "mede mo asofodie no mama mo se achedie som ajuma; na obi foforo biara a obebepene ho no, wonku no."
    }
]

meta = []
for v in voices:
    vid = v["id"]
    raw_a = load_audio(v["source_wav"], sr)[0].numpy()
    if len(raw_a) > int(7.0 * sr):
        raw_a = raw_a[:int(7.0 * sr)]
    norm_a = normalize_loudness(raw_a)
    w_path = os.path.join(out_dir, v["file"])
    sf.write(w_path, norm_a, sr)
    txt_path = os.path.join(out_dir, vid + ".txt")
    with open(txt_path, "w", encoding="utf-8") as tf:
        tf.write(v["prompt_text_universal"])
    v_clean = {k: v[k] for k in v if k != "source_wav"}
    meta.append(v_clean)

with open(os.path.join(out_dir, "voices.json"), "w", encoding="utf-8") as jf:
    json.dump(meta, jf, indent=2, ensure_ascii=False)

print("All 6 clean preset voices written successfully to", out_dir)
