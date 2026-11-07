"""
TTS synthesis for the Open TTS Leaderboard (ZONOS3 Mini backend, stage 1).

ZONOS3 Mini (Zyphra): runtime https://github.com/Zyphra/ZONOS3 (revision pinned in submit_jobs.sh), weights
Zyphra/ZONOS3_Mini_release_ckpts (release LM lm/rate30_20k_r2grpo, KEEP-2 decoder). The job starts the
runtime's own server (servers/sgl/scripts/serve.sh); this script sends `--batch_size` concurrent
POST /v2/generate requests per batch, and per-sample time = batch wall time / batch size.

Voice cloning is speaker-embedding only (`speaker_mode: "embedding"`): the server embeds each prompt clip
with the ownmect_s2 MECT-B2 speaker encoder; `prompt_text` is not used. This is ZTTS2-Eval's
`zonos3-rate30-20k-r2grpo_embedding` protocol: release sampling defaults (left to the server), seed 0 and a
1350-frame cap on every request, server text normalization for en zh fr de es it pt ja ko, and every other
language sent as en_us with normalization off.

Text frontend (inside the timed request): in languages without server normalization, numbers are spelled out with
num2words (none for el, hi), as the server would otherwise hand the digits to the model as typed. The manifest keeps the
dataset text.
"""

import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
from num2words import num2words
from tqdm import tqdm

from run_eval_utils import (
    add_common_args,
    load_done_entries,
    load_tts_dataset,
    manifest_entry,
    open_manifest,
    output_paths,
    pending_batches,
    print_next_steps,
    set_seed,
    warm_up,
    wav_rel_path,
    write_entry,
)

# Leaderboard language -> ZONOS3 server language (model/src/zonos3/registries/languages.json). Any other
# language is sent as en_us with text normalization off, as ZTTS2-Eval's generate_zonos3.py does.
SERVER_LANGUAGE = {"en": "en_us", "zh": "cmn", "fr": "fr_fr", "de": "de", "es": "es", "it": "it",
                   "pt": "pt_br", "ja": "ja", "ko": "ko"}
EMPTY_AUDIO_SECONDS = 0.1  # an empty response is written as this much silence, so it scores as a full miss

# Languages without server normalization whose numbers num2words reads (the server would hand digits to the model as
# typed): decimal separator, and the word for "%" (Turkish says it before the number).
NUMBER_DECIMAL = {"ru": ",", "nl": ",", "pl": ",", "cs": ",", "ro": ",", "fi": ",", "tr": ",", "id": ",", "vi": ",",
                  "ar": ".", "th": "."}
NUMBER_PERCENT = {"ru": "процентов", "nl": "procent", "pl": "procent", "cs": "procent", "ro": "la sută", "fi": "prosenttia",
                  "tr": "yüzde", "id": "persen", "vi": "phần trăm", "ar": "بالمئة", "th": "เปอร์เซ็นต์"}
# num2words' decimal reading is wrong for these ("3,5" -> vi "ba phẩy năm mươi"): integer part, this word, fraction digits
DECIMAL_WORD = {"tr": "virgül", "vi": "phẩy", "ar": "فاصلة", "th": "จุด"}
# an optional "%" before (Turkish writing) or after the number: thousand groups split by spaces, or digits with "." /
# "," inside (one separator: a decimal or thousands; more: a date or version, read part by part)
NUMBER = re.compile(r"(%\s?)?(?<![\w.,])(\d{1,3}(?:[ \u00a0]\d{3})+(?!\d)|\d+(?:[.,]\d+)*)(\s?%)?")


def number_words(match, lang):
    """One number (thousand groups, decimal, optional %) in words; the match unchanged if num2words fails."""
    raw, percent = match.group(2), match.group(1) or match.group(3)
    dec = NUMBER_DECIMAL[lang]
    try:
        if re.fullmatch(r"\d+", raw) or any(c in raw for c in " \u00a0") or re.fullmatch(r"\d+[.,]\d{3}", raw) and dec not in raw:
            words = num2words(int(re.sub(r"\D", "", raw)), lang=lang)  # an integer (the other separator groups thousands)
        elif dec in raw and raw.count(dec) == 1 and not re.search(r"[.,]", raw.replace(dec, "", 1)):
            whole, frac = raw.split(dec)
            if lang in DECIMAL_WORD:
                zeros = len(frac) - len(frac.lstrip("0"))
                words = " ".join([num2words(int(whole), lang=lang), DECIMAL_WORD[lang]]
                                 + [num2words(0, lang=lang)] * zeros + ([num2words(int(frac), lang=lang)] if frac.lstrip("0") else []))
            else:
                words = num2words(float(f"{whole}.{frac}"), lang=lang)
        else:  # the other separator (not a thousands group), or several: a date, version or ordinal, part by part
            words = " ".join(num2words(int(part), lang=lang) for part in re.split(r"[.,]", raw))
    except (NotImplementedError, OverflowError, ValueError):
        return match.group(0)
    if percent:
        words = f"{NUMBER_PERCENT[lang]} {words}" if lang == "tr" else f"{words} {NUMBER_PERCENT[lang]}"
    return words


def spell_numbers(text, lang):
    """Digits -> words for a language without server normalization (unchanged where num2words has no reading)."""
    if lang not in NUMBER_DECIMAL:
        return text

    def replace(m):
        before, after = text[m.start() - 1 : m.start()], text[m.end() : m.end() + 1]
        return (" " if before.isalnum() else "") + number_words(m, lang) + (" " if after.isalnum() else "")
    return NUMBER.sub(replace, text)


def http(url, payload=None, timeout=60):
    """GET (payload None) or POST JSON; returns (body bytes, response headers)."""
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"} if data is not None else {}
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=headers), timeout=timeout) as resp:
        return resp.read(), resp.headers


def process_alive(pid):
    """False once `pid` has exited (a zombie counts as exited)."""
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def wait_for_server(base_url, timeout, server_pid, server_log=None):
    """Poll GET /v2/health; fail fast if the server process dies or its log shows a traceback (a scheduler or tokenizer
    subprocess that crashes leaves the API process up but never healthy). The first start JIT-compiles kernels."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            http(f"{base_url}/v2/health", timeout=10)
            return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            pass
        if server_pid and not process_alive(server_pid):
            raise RuntimeError("ZONOS3 server exited before it became healthy (its log tail follows).")
        if server_log and os.path.exists(server_log):
            with open(server_log, errors="replace") as f:
                if "Traceback (most recent call last)" in f.read():
                    raise RuntimeError("ZONOS3 server raised during startup (its log tail follows).")
        if time.monotonic() > deadline:
            raise RuntimeError(f"ZONOS3 server not healthy after {timeout}s.")
        time.sleep(10)


def synthesize(base_url, payload, timeout, retries):
    """POST /v2/generate -> (float32 mono audio, sample rate). Retries 5xx/connection errors, raises on 4xx."""
    for attempt in range(retries + 1):
        try:
            body, headers = http(f"{base_url}/v2/generate", payload, timeout=timeout)
            return np.frombuffer(body, dtype="<f4").copy(), int(headers.get("X-Audio-Sample-Rate", 44100))
        except urllib.error.HTTPError as e:
            if e.code < 500 or attempt == retries:
                raise RuntimeError(f"/v2/generate HTTP {e.code}: {e.read()[:500]!r}") from e
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if attempt == retries:
                raise
        time.sleep(2.0 * 2**attempt)


def main(args):
    if not args.voice_clone:
        raise SystemExit("The ZONOS3 backend evaluates speaker-embedding cloning only; run with --voice_clone.")
    if args.ttfa_probe:
        raise SystemExit("--ttfa_probe is not implemented for the ZONOS3 backend.")
    set_seed(args.seed)
    language = args.language or args.split
    server_language = SERVER_LANGUAGE.get(language, "en_us")
    text_normalization = language in SERVER_LANGUAGE

    wait_for_server(args.server_url, args.server_timeout, args.server_pid, args.server_log)
    caps = json.loads(http(f"{args.server_url}/v2/capabilities")[0])
    if "embedding" not in caps["speaker_modes"]:
        raise SystemExit(f"The served checkpoint has no embedding speaker mode: {caps['speaker_modes']}")
    served = json.loads(http(f"{args.server_url}/v1/models")[0])["data"][0]["root"]
    print(f"ZONOS3 server: model={served} codec={caps['audio_codec']} decoder={json.dumps(caps['decoder'])} "
          f"speaker_embedding_codec={caps['speaker_embedding_codec']} max_tokens={caps['max_tokens']}")
    print(f"language={language} -> server language={server_language}, text_normalization={text_normalization}")

    paths = output_paths(args, "_voice_clone")
    os.makedirs(os.path.join(paths.output_dir, "prompts"), exist_ok=True)
    dataset = load_tts_dataset(args, extra_columns=("prompt_audio",))
    executor = ThreadPoolExecutor(max_workers=args.batch_size)

    def generate_tts(batch):
        """Synthesize a minibatch as concurrent requests; returns its manifest entries."""
        texts = list(batch[args.text_column])
        payloads, prompt_rel_paths = [], []
        # Prompt clips are written under results/ (untimed) so stage 3 (SIM) reads the clip the server embedded.
        for sample_id, text, prompt_audio in zip(batch["id"], texts, batch["prompt_audio"]):
            prel = os.path.join(paths.dataset_dir_name, "prompts", f"prompt_{sample_id}.wav")
            ppath = os.path.join(paths.model_dir, prel)
            if not os.path.exists(ppath):
                sf.write(ppath, np.asarray(prompt_audio["array"], dtype=np.float32), prompt_audio["sampling_rate"])
            with open(ppath, "rb") as f:
                speaker_audio = base64.b64encode(f.read()).decode("ascii")
            payloads.append({
                "text": text, "language": server_language, "text_normalization": text_normalization,
                "speaker_mode": "embedding", "speaker_audio_base64": speaker_audio,
                "speaker_audio_name": os.path.basename(ppath),
                "seed": args.seed, "max_tokens": args.max_tokens, "stream": False, "format": "pcm",
            })
            prompt_rel_paths.append(prel)

        def request(payload):
            payload = dict(payload, text=spell_numbers(payload["text"], language))
            return synthesize(args.server_url, payload, args.request_timeout, args.retries)

        start = time.perf_counter()
        outputs = list(executor.map(request, payloads))
        per_sample_time = (time.perf_counter() - start) / len(payloads)

        entries = []
        for sample_id, text, prel, (audio, sample_rate) in zip(batch["id"], texts, prompt_rel_paths, outputs):
            if audio.size == 0:
                print(f"WARNING: empty audio for sample {sample_id}; writing {EMPTY_AUDIO_SECONDS}s of silence.")
                audio = np.zeros(int(EMPTY_AUDIO_SECONDS * sample_rate), dtype=np.float32)
            rel = wav_rel_path(paths.dataset_dir_name, sample_id)
            sf.write(os.path.join(paths.model_dir, rel), audio, sample_rate)
            entries.append(manifest_entry(rel, len(audio) / sample_rate, per_sample_time, text, prel))
        return entries

    done = load_done_entries(paths.manifest_path, args.resume)
    starts = pending_batches(dataset, args.batch_size, done, paths.dataset_dir_name)
    warm_up(generate_tts, dataset, starts, args.batch_size, args.warmup_steps)

    with open_manifest(paths.manifest_path, done) as manifest_file:
        for batch_start in tqdm(starts, desc="Generating"):
            for entry in generate_tts(dataset[batch_start : batch_start + args.batch_size]):
                write_entry(manifest_file, entry)
            manifest_file.flush()
    executor.shutdown()
    print("Results saved at path:", os.path.abspath(paths.manifest_path))
    print_next_steps(True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser, batch_size=16, voice_clone=True)
    parser.set_defaults(seed=0)  # ZTTS2-Eval protocol: seed 0 on every request
    parser.add_argument("--model_id", default="Zyphra/ZONOS3-Mini", help="Leaderboard model id (names the results folder).")
    parser.add_argument("--language", default=None, help="Language code of the split (default: --split).")
    parser.add_argument("--server_url", default="http://127.0.0.1:1919", help="ZONOS3 server (serve.sh's port).")
    parser.add_argument("--server_pid", type=int, default=0, help="Server PID; the startup wait fails fast if it exits.")
    parser.add_argument("--server_log", default=None, help="Server log; the startup wait fails fast on a traceback in it.")
    parser.add_argument("--server_timeout", type=int, default=3600, help="Seconds to wait for /v2/health.")
    parser.add_argument("--request_timeout", type=int, default=900, help="Seconds per /v2/generate request.")
    parser.add_argument("--retries", type=int, default=4, help="Retries per request on 5xx / connection errors.")
    parser.add_argument("--max_tokens", type=int, default=1350, help="Frame cap per request (15 Hz: 1350 = 90 s).")
    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
