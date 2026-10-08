"""
cer_watch.py — Side CER evaluator: watches a run's checkpoints and scores each one
with omniASR-CTC-300M, without touching training.

For every new <out_dir>/step_XXXXXXX (plus the base model and the real recordings
once): synthesise a fixed eval set — per language, N held-out target sentences,
each generated WITH a voice prompt and WITHOUT one — transcribe with omniASR
CTC 300M (sherpa-onnx export, on CPU so it takes no GPU) and score CER:

  cer_native     hyp vs the native-orthography transcript (omniASR writes native)
  cer_universal  both converted to africa-g2p universal first (forgives spelling
                 variants that sound the same)

Per-utterance CER is capped at 100; overall = macro average over all languages
(Dangme/Sehwi have no omniASR language code but the CTC model transcribes them fine).
"real" = omniASR on the actual recordings: the floor any TTS can reach.

Results: <out_dir>/cer_results.jsonl (one row per tag × mode) and
         <out_dir>/cer/<tag>/rows.json (every utterance with its hypothesis).
Generated wavs already on disk are reused, so a tag can be re-scored cheaply by
deleting its rows from cer_results.jsonl.

    python train/cer_watch.py --config train/config_ghana_noprompt.yaml
"""

import argparse
import json
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path

FINETUNE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(FINETUNE_DIR.parent))
sys.path.insert(0, str(FINETUNE_DIR))

import multiprocessing as mp

import soundfile as sf
import torch
import yaml
from africa_g2p import UNIVERSAL, convert_lang
from transformers import AutoTokenizer

import ghana_audiodit.audiodit  # noqa: F401  (registers AudioDiT with transformers)
from ghana_audiodit.audiodit import AudioDiTModel
from lora_utils import load_lora
from noprompt_test import frames_per_char, generate
from sample import synth
from ghana_audiodit.utils import normalize_text

ASR_DIR = os.environ.get("OMNIASR_SHERPA_DIR", "sherpa-onnx-omnilingual-asr-1600-languages-300M-ctc-2025-11-12")
ASR_WORKERS, ASR_THREADS = 4, 2   # graph doesn't parallelise intra-op: scale with processes
# omniASR language codes; None = no code (CTC model transcribes without one)
SPECIAL = {"twi": "aka_Latn", "ada": None, "sfw": None, "ffm": "fub_Latn", "eng": "eng_Latn"}


def asr_code(lang: str):
    iso = lang.rsplit("_", 1)[-1].lower()
    return SPECIAL.get(iso, f"{iso}_Latn")


def norm(text: str) -> str:
    out = []
    for ch in text.upper():
        if ch.isspace():
            out.append(" ")
        elif unicodedata.category(ch)[0] in ("L", "N", "M"):
            out.append(ch)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def cer(ref: str, hyp: str) -> float:
    if not ref:
        return 0.0 if not hyp else 100.0
    prev = list(range(len(hyp) + 1))
    for i in range(1, len(ref) + 1):
        cur = [i] + [0] * len(hyp)
        for j in range(1, len(hyp) + 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ref[i - 1] != hyp[j - 1]))
        prev = cur
    return min(prev[-1] / len(ref) * 100, 100.0)


def to_univ(text: str, lang: str) -> str:
    code = lang.rsplit("_", 1)[-1]
    if code == "eng":
        return text
    try:
        return convert_lang(text, code, UNIVERSAL)
    except Exception:
        return text


def build_eval_set(val_manifest: str, per_lang: int, seed: int = 0, draw: int = 3,
                   langs: list[str] | None = None):
    """Fixed draw of `draw` sentences per language; the first `per_lang` are used.
    Each item keeps its index in the full draw (wav file names), so evaluating a
    smaller subset re-uses — and stays comparable with — earlier audio."""
    by_lang: dict[str, list[dict]] = {}
    for line in open(val_manifest, encoding="utf-8"):
        m = json.loads(line)
        by_lang.setdefault(m["lang"], []).append(m)
    rng = random.Random(seed)
    items = []
    for lang in sorted(by_lang):
        if langs and lang not in langs:
            continue
        clips = by_lang[lang]
        prompts = [c for c in clips if 35 <= c["frames"] <= 117]
        if not prompts:
            continue
        p = rng.choice(prompts)
        targets = [c for c in clips if c["id"] != p["id"] and len(c.get("text_native", c["text"])) >= 20]
        for k, t in enumerate(rng.sample(targets, min(draw, len(targets)))):
            items.append({"lang": lang, "prompt": p, "target": t, "n": len(items), "k": k})
    return [it for it in items if it["k"] < per_lang]


_rec = None


def _init_asr():
    global _rec
    import sherpa_onnx
    _rec = sherpa_onnx.OfflineRecognizer.from_omnilingual_asr_ctc(
        model=f"{ASR_DIR}/model.onnx", tokens=f"{ASR_DIR}/tokens.txt",
        num_threads=ASR_THREADS, decoding_method="greedy_search")


def _transcribe(path: str) -> str:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    stream = _rec.create_stream()
    stream.accept_waveform(sr, audio.mean(axis=1))
    _rec.decode_stream(stream)
    return (stream.result.text or "").strip()


def run_asr(jobs: list[dict], work: Path) -> list[str]:
    """CTC head takes no language hint; job["code"] is kept only for the record."""
    with mp.get_context("spawn").Pool(ASR_WORKERS, initializer=_init_asr) as pool:
        return pool.map(_transcribe, [j["wav"] for j in jobs])


def score(items, hyps, mode: str, tag: str, step: int, secs: float) -> tuple[dict, list[dict]]:
    rows, per_n, per_u = [], {}, {}
    for it, hyp in zip(items, hyps):
        lang, ref = it["lang"], it["target"].get("text_native", it["target"]["text"])
        cn = cer(norm(ref), norm(hyp))
        cu = cer(norm(to_univ(ref, lang)), norm(to_univ(hyp, lang)))
        rows.append({"lang": lang, "mode": mode, "id": it["target"]["id"], "ref": ref, "hyp": hyp,
                     "cer_native": round(cn, 2), "cer_universal": round(cu, 2)})
        per_n.setdefault(lang, []).append(cn)
        per_u.setdefault(lang, []).append(cu)
    avg = lambda d: {k: round(sum(v) / len(v), 2) for k, v in sorted(d.items())}
    ln, lu = avg(per_n), avg(per_u)
    rec = {"tag": tag, "step": step, "mode": mode,
           "cer_native": round(sum(ln.values()) / max(len(ln), 1), 2),
           "cer_universal": round(sum(lu.values()) / max(len(lu), 1), 2),
           "n_langs": len(ln), "n_utts": len(rows), "secs": round(secs),
           "per_lang_native": ln, "per_lang_universal": lu, "time": time.strftime("%H:%M")}
    return rec, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--per_lang", type=int, default=3)
    ap.add_argument("--draw", type=int, default=3, help="sentences drawn per language (first per_lang used)")
    ap.add_argument("--langs", nargs="*", default=None, help="restrict to these languages (folder names)")
    ap.add_argument("--name", default="cer", help="results file <name>_results.jsonl and work dir <name>/")
    ap.add_argument("--every", type=int, default=2000, help="score checkpoints whose step is a multiple")
    ap.add_argument("--poll", type=int, default=60)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    s_cfg = cfg.get("samples", {})
    out_dir = Path(cfg["training"]["output_dir"])
    results = out_dir / f"{args.name}_results.jsonl"
    items = build_eval_set(cfg["data"]["val_manifest"], args.per_lang, draw=max(args.draw, args.per_lang),
                           langs=args.langs)
    fpc = frames_per_char(cfg["data"]["train_manifest"])
    print(f"[cer] eval set: {len(items)} sentences × 2 modes over {len({i['lang'] for i in items})} languages",
          flush=True)

    def done_tags():
        """A tag is done once every mode it needs is scored ("real": 1, models: 2)."""
        if not results.exists():
            return set()
        modes: dict[str, set] = {}
        for l in open(results, encoding="utf-8"):
            r = json.loads(l)
            modes.setdefault(r["tag"], set()).add(r["mode"])
        return {t for t, m in modes.items() if len(m) >= (1 if t == "real" else 2)}

    def append(rec):
        with open(results, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"[cer] {rec['time']} {rec['tag']:14s} {rec['mode']:9s} CER native={rec['cer_native']:6.2f}  "
              f"universal={rec['cer_universal']:6.2f}  ({rec['n_langs']} langs, {rec['secs']}s)", flush=True)

    tokenizer = None
    while True:
        done = done_tags()
        todo = []
        if "real" not in done:
            todo.append(("real", 0, None))
        if "base" not in done:
            todo.append(("base", 0, None))
        for d in sorted(out_dir.glob("step_*")):
            if d.name not in done and (d / "training_state.pt").exists() \
                    and int(d.name.split("_")[1]) % args.every == 0:
                todo.append((d.name, int(d.name.split("_")[1]), d))
        if not todo:
            if (out_dir / "merged").exists():
                print("[cer] training finished and all checkpoints scored — exiting", flush=True)
                return
            time.sleep(args.poll)
            continue

        tag, step, ckpt = todo[0]
        work = out_dir / args.name / tag
        work.mkdir(parents=True, exist_ok=True)
        t0 = time.time()

        if tag == "real":   # omniASR on the actual target recordings
            jobs = [{"wav": it["target"]["wav"], "code": asr_code(it["lang"])} for it in items]
            rec, rows = score(items, run_asr(jobs, work), "real", tag, 0, time.time() - t0)
            json.dump(rows, open(work / "rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
            append(rec)
            continue

        wav_path = lambda mode, n, lang: work / f"{mode}_{n:03d}_{lang}.wav"
        have_all = all(wav_path(m, it["n"], it["lang"]).exists()
                       for m in ("prompt", "noprompt") for it in items)
        model = None
        if not have_all:
            model = AudioDiTModel.from_pretrained(cfg["model"]["model_dir"]).to("cuda:0")
            model.vae.to_half()
            if ckpt is not None:
                load_lora(model, ckpt)
            model.eval()
            tokenizer = tokenizer or AutoTokenizer.from_pretrained(model.config.text_encoder_model)
            sr = model.config.sampling_rate
            max_frames = int(model.config.max_wav_duration * sr // model.config.latent_hop)

        all_rows = []
        for mode in ("prompt", "noprompt"):
            t1 = time.time()
            jobs = []
            for it in items:
                n, p, t, lang = it["n"], it["prompt"], it["target"], it["lang"]
                path = wav_path(mode, n, lang)
                if path.exists():   # reuse audio from an earlier scoring pass
                    jobs.append({"wav": str(path), "code": asr_code(lang)})
                    continue
                torch.manual_seed(1234 + n)
                if mode == "prompt":
                    wav = synth(model, tokenizer, p["wav"], p["text"], t["text"],
                                nfe=s_cfg.get("nfe", 16), cfg_strength=s_cfg.get("cfg_strength", 4.0),
                                guidance_method=s_cfg.get("guidance_method", "apg"))
                else:
                    frames = min(max(int(fpc[lang] * len(normalize_text(t["text"]).replace(" ", ""))), 12),
                                 max_frames)
                    wav = generate(model, tokenizer, t["text"], frames, 1234 + n, s_cfg)
                sf.write(path, wav, sr)
                jobs.append({"wav": str(path), "code": asr_code(lang)})
            rec, rows = score(items, run_asr(jobs, work), mode, tag, step, time.time() - t1)
            all_rows += rows
            append(rec)
        json.dump(all_rows, open(work / "rows.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        if model is not None:
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
