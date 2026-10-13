import sys
import os
import glob
import json
import time
import sherpa_onnx
import soundfile as sf
import torch
from africa_g2p import UNIVERSAL, convert_lang
from transformers import AutoTokenizer

sys.path.insert(0, "/mnt/volume_d2wey28/projects/longcat-audiodit-ghana/ghana-audiodit-repo")
sys.path.insert(0, "/mnt/volume_d2wey28/projects/longcat-audiodit-ghana/ghana-audiodit-repo/training")
import ghana_audiodit.audiodit
from ghana_audiodit.audiodit import AudioDiTModel
from lora_utils import load_lora
from sample import synth

asr_dir = "/mnt/volume_d2wey28/models/sherpa/sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12"
rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
    model=f"{asr_dir}/model.onnx",
    tokens=f"{asr_dir}/tokens.txt",
    num_threads=4,
    decoding_method="greedy_search"
)

def norm(text):
    return " ".join("".join(c for c in text.upper() if c.isalnum() or c.isspace()).split())

def cer(ref, hyp):
    if not ref:
        return 0.0 if not hyp else 100.0
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return min(prev[-1] / len(ref) * 100, 100.0)

def transcribe(wav_path):
    audio, sr = sf.read(wav_path, dtype="float32", always_2d=True)
    stream = rec.create_stream()
    stream.accept_waveform(sr, audio.mean(axis=1))
    rec.decode_stream(stream)
    return (stream.result.text or "").strip()

# 1. Load clean test pairs
val_path = "/mnt/volume_d2wey28/projects/longcat-audiodit-ghana/latents/val_universal.jsonl"
clips = []
for line in open(val_path):
    c = json.loads(line)
    if c.get("lang") == "Asante_Twi_twi" and 35 <= c.get("frames", 0) <= 120:
        clips.append(c)

prompt_clip = clips[0]
target_clips = clips[1:5]  # 4 evaluation targets

print("Prompt ID:", prompt_clip["id"])
for i, t in enumerate(target_clips):
    print("Target", i+1, ":", t["text"][:60], "...")

# 2. Ground Truth Real Speech Floor
real_cers = []
for t in target_clips:
    h = transcribe(t["wav"])
    c = cer(norm(t["text"]), norm(convert_lang(h, "twi", UNIVERSAL)))
    real_cers.append(c)
floor_cer = sum(real_cers) / len(real_cers)
print(f"\nREAL SPEECH FLOOR CER (omniASR on real audio): {floor_cer:.2f}%\n")

# 3. Checkpoints to evaluate
ckpts = ["base_model"]
out_dir = "/mnt/volume_d2wey28/projects/longcat-audiodit-ghana/checkpoints/lora_r64_twi_prompted"
for s in ["step_0009000", "step_0010000", "step_0011000", "step_0012000", "step_0013000"]:
    if os.path.exists(f"{out_dir}/{s}"):
        ckpts.append(s)

print("Checkpoint       | Universal CER   | Native CER      | Time")
print("-" * 60)

for c_name in ckpts:
    t0 = time.time()
    m = AudioDiTModel.from_pretrained("meituan-longcat/LongCat-AudioDiT-1B").to("cuda:0")
    m.vae.to_half()
    if c_name != "base_model":
        load_lora(m, f"{out_dir}/{c_name}")
    m.eval()
    tok = AutoTokenizer.from_pretrained(m.config.text_encoder_model)

    u_cers, n_cers = [], []
    for i, t in enumerate(target_clips):
        w = synth(m, tok, prompt_clip["wav"], prompt_clip["text"], t["text"])
        tmp_w = f"/tmp/eval_{c_name}_{i}.wav"
        sf.write(tmp_w, w, 24000)
        hyp = transcribe(tmp_w)

        ref_u = norm(t["text"])
        hyp_u = norm(convert_lang(hyp, "twi", UNIVERSAL))
        ref_n = norm(t.get("text_native", t["text"]))
        hyp_n = norm(hyp)

        u_cers.append(cer(ref_u, hyp_u))
        n_cers.append(cer(ref_n, hyp_n))

    del m
    torch.cuda.empty_cache()

    dt = time.time() - t0
    avg_u = sum(u_cers) / len(u_cers)
    avg_n = sum(n_cers) / len(n_cers)
    print(f"{c_name:16s} | {avg_u:13.2f}% | {avg_n:13.2f}% | {dt:6.1f}s")
