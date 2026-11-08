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
import dataclasses
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import soundfile as sf
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
    from num2words import num2words  # client venv only (the in-process TTFA worker runs under the runtime's venv)

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


def stream_synthesize(base_url, payload, timeout):
    """POST /v2/generate as a stream, read to the end -> (float32 audio, sample rate, perf_counter at the first audio
    bytes, reads)."""
    data = json.dumps(dict(payload, stream=True)).encode("utf-8")
    request = urllib.request.Request(f"{base_url}/v2/generate", data=data, headers={"Content-Type": "application/json"})
    first, parts = None, []
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        sample_rate = int(resp.headers.get("X-Audio-Sample-Rate", 44100))
        while chunk := resp.read1(1 << 16):
            if first is None:
                first = time.perf_counter()
            parts.append(chunk)
    body = b"".join(parts)
    return np.frombuffer(body[: len(body) // 4 * 4], dtype="<f4"), sample_rate, first, len(parts)


def ttfa_probe(args, language, server_language, text_normalization, caps):
    """Time to first audio at batch size 1 (scripts/ttfa_probe.py): one streaming request at a time with the eval's
    seed and frame cap, prefix-streamed (stream_decode null = the server's prefix default for this codec). Without
    --voice_clone the voice is the server's default voice --voice, in speaker_mode "embedding" like every eval request
    (the release checkpoint's default mode is "reference": the voice's audio codes as a prompt prefix, a longer prefill
    and another system). Every response is read to the end (no early stop):
    the server keeps generating a request whose client disconnected, which would delay the next probe request."""
    from ttfa_probe import run_probe, sample_indices  # injected by the TTFA job, like every backend's probe

    voice_id = None
    if not args.voice_clone:  # default voices are listed by a hashed id; --voice may name the label
        speakers = json.loads(http(f"{args.server_url}/v2/speakers")[0])["speakers"]
        voice_id = next((s["id"] for s in speakers if args.voice in (s["id"], s.get("label"))), None)
        if voice_id is None:
            raise SystemExit(f"--voice {args.voice!r} is not one of the server's speakers: {[s.get('label') for s in speakers]}")
    mode_suffix = "_voice_clone" if args.voice_clone else ""
    dataset = load_tts_dataset(args, extra_columns=("prompt_audio",) if args.voice_clone else ())
    payload = {"language": server_language, "text_normalization": text_normalization, "seed": args.seed,
               "max_tokens": args.max_tokens, "format": "pcm", "laci": args.laci}
    if args.stream_codebooks is not None:  # early partial-frame decode (null = the server default)
        payload["stream_codebooks"] = args.stream_codebooks
    jobs = []
    with tempfile.TemporaryDirectory(prefix="ttfa_prompts_") as prompt_dir:
        for i in sample_indices(args.ttfa_probe, len(dataset)):
            row = dataset[i]
            job = dict(payload, text=row[args.text_column])
            if args.voice_clone:  # the prompt is written as the eval writes it; its upload is part of the request
                path = os.path.join(prompt_dir, f"prompt_{row['id']}.wav")
                sf.write(path, np.asarray(row["prompt_audio"]["array"], dtype=np.float32), row["prompt_audio"]["sampling_rate"])
                with open(path, "rb") as f:
                    job.update(speaker_mode="embedding", speaker_audio_base64=base64.b64encode(f.read()).decode("ascii"),
                               speaker_audio_name=os.path.basename(path))
            else:
                job.update(speaker_embedding_id=voice_id, speaker_mode="embedding")
            jobs.append(job)
    reads = []

    def gen(job):
        job = dict(job, text=spell_numbers(job["text"], language))
        audio, sample_rate, first, n_reads = stream_synthesize(args.server_url, job, args.request_timeout)
        reads.append(n_reads)
        return audio, sample_rate, first

    paths = output_paths(args, mode_suffix)
    run_probe(gen, jobs, model_id=args.model_id, dataset_path=args.dataset_path, dataset_config=args.dataset,
              split=args.split, out_dir=paths.model_dir, mode_suffix=mode_suffix, n=args.ttfa_probe,
              extra={"engine": "zonos3-sgl", "batch_size": 1, "voice": None if args.voice_clone else args.voice,
                     "speaker_mode": "embedding", "seed": args.seed,
                     "max_tokens": args.max_tokens, "stream_decode": "server default (prefix)",
                     "text_normalization": text_normalization, "laci": args.laci,
                     "stream_codebooks": args.stream_codebooks if args.stream_codebooks is not None else "server default",
                     "decoder": caps["decoder"], "http_reads_per_utterance": reads})


def ttfa_in_process(args, language):
    """--in_process: the TTFA probe without the server -- scripts/ttfa_probe.py timing a generator in the model's own
    process, as the leaderboard's in-process engines are measured (faster-qwen3-tts, audio.cpp). This parent (the
    client venv) writes the probe's texts, number-spelled as the server probe sends them, and runs this script as the
    worker under the runtime's interpreter after the runtime's env.sh (serve.sh's environment)."""
    if args.voice_clone or args.laci:
        raise SystemExit("--in_process probes the default voice without LACI (--no-voice_clone, no --laci)")
    from ttfa_probe import sample_indices

    dataset = load_tts_dataset(args)
    texts = [spell_numbers(dataset[i][args.text_column], language) for i in sample_indices(args.ttfa_probe, len(dataset))]
    with tempfile.TemporaryDirectory(prefix="ttfa_in_process_") as tmp:
        jobs = os.path.join(tmp, "texts.json")
        with open(jobs, "w") as f:
            json.dump(texts, f)
        runtime = os.path.abspath(args.runtime_dir)
        worker = [os.path.join(runtime, ".venv", "bin", "python"), os.path.abspath(__file__), *sys.argv[1:], f"--ttfa_worker={jobs}"]
        env_sh = shlex.quote(os.path.join(runtime, "servers", "sgl", "scripts", "env.sh"))
        # serve.sh also puts the speaker embedder on CUDA (its default steering bundle needs it there)
        subprocess.run(["bash", "-c", f"source {env_sh} && export ZONOS3_SPEAKER_EMBEDDER_DEVICE=cuda && exec {shlex.join(worker)}"],
                       check=True)


def ttfa_worker(args, server_language, text_normalization):
    """The in-process probe's worker (the runtime's venv): TTSLLM with serve.sh's engine settings, the default voice
    embedded as the server embeds it, and every text streamed by TTSLLM.generate_stream -- the server's request (silence
    prefill, registry sampling, seed, frame cap, prefix stream with the request's stream_codebooks) decoded by the
    server's own vocoder code, without the frontend, the tokenizer process and their IPC."""
    from pathlib import Path

    from ttfa_probe import run_probe
    from zonos3.generation import SAMPLE_RATE
    from zonos3.server.voices import scan_default_voices, voice_preview
    from zonos3.tts.audio import _decode_wav_bytes
    from zonos3_sgl.message.tts import TTSSamplingParams
    from zonos3_sgl.tts.llm import TTSLLM

    with open(args.ttfa_worker) as f:
        texts = json.load(f)
    tts = TTSLLM(model_path=args.lm, attention_backend="fa", cache_type="naive", max_running_req=2,
                 cuda_graph_max_bs=2, max_extend_tokens=256, memory_ratio=0.5)
    # serve.sh's --tts-default-voices-dir; the server embeds a default voice from its WAV transcode (voice_preview), not
    # the file itself -- another embedding, hence other tokens
    voices = scan_default_voices(Path(args.runtime_dir) / "assets" / "default_voices", {}).values()
    entry = next((v for v in voices if args.voice in (v.speaker_id, v.label)), None)
    if entry is None:
        raise SystemExit(f"--voice {args.voice!r} is not a default voice: {sorted(v.label for v in voices)}")
    voice = tts.embed_speaker(*_decode_wav_bytes(voice_preview(entry)[0]))
    sampling = TTSSamplingParams(seed=args.seed)
    if args.stream_codebooks is not None:
        sampling = dataclasses.replace(sampling, stream_codebooks=args.stream_codebooks)
    print(f"in-process: TTSLLM {args.lm}, codec {tts.engine.model_config.audio_codec}, voice {args.voice}, "
          f"language={server_language}, text_normalization={text_normalization}, stream_codebooks={sampling.stream_codebooks}")

    def gen(text, early_stop):
        first, samples = None, 0
        stream = tts.generate_stream(text, sampling, max_tokens=args.max_tokens, language=server_language,
                                     text_normalization=text_normalization, speaker_embedding=voice)
        for chunk in stream:
            if first is None:
                first = time.perf_counter()
            samples += len(chunk)
            if early_stop:  # runs the request to its end undecoded (the scheduler has no abort)
                stream.close()
                break
        return samples / SAMPLE_RATE, first

    paths = output_paths(args)
    run_probe(gen, texts, model_id=args.model_id, dataset_path=args.dataset_path, dataset_config=args.dataset,
              split=args.split, out_dir=paths.model_dir, n=-1,
              extra={"engine": "zonos3-sgl in-process (TTSLLM.generate_stream)", "batch_size": 1, "voice": args.voice,
                     "speaker_mode": "embedding", "seed": args.seed, "max_tokens": args.max_tokens,
                     "stream_decode": "prefix", "text_normalization": text_normalization,
                     "stream_codebooks": sampling.stream_codebooks, "lm": args.lm,
                     "audio_codec": tts.engine.model_config.audio_codec})


def main(args):
    if not args.voice_clone and not args.ttfa_probe:
        raise SystemExit("The ZONOS3 backend evaluates speaker-embedding cloning only; run with --voice_clone.")
    set_seed(args.seed)
    language = args.language or args.split
    server_language = SERVER_LANGUAGE.get(language, "en_us")
    text_normalization = language in SERVER_LANGUAGE and args.text_normalization
    if args.ttfa_worker:
        ttfa_worker(args, server_language, text_normalization)
        return
    if args.ttfa_probe and args.in_process:
        ttfa_in_process(args, language)
        return

    wait_for_server(args.server_url, args.server_timeout, args.server_pid, args.server_log)
    caps = json.loads(http(f"{args.server_url}/v2/capabilities")[0])
    if "embedding" not in caps["speaker_modes"]:
        raise SystemExit(f"The served checkpoint has no embedding speaker mode: {caps['speaker_modes']}")
    served = json.loads(http(f"{args.server_url}/v1/models")[0])["data"][0]["root"]
    print(f"ZONOS3 server: model={served} codec={caps['audio_codec']} decoder={json.dumps(caps['decoder'])} "
          f"speaker_embedding_codec={caps['speaker_embedding_codec']} max_tokens={caps['max_tokens']}")
    print(f"language={language} -> server language={server_language}, text_normalization={text_normalization}")
    if args.ttfa_probe:
        ttfa_probe(args, language, server_language, text_normalization, caps)
        return

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
                "seed": args.seed, "max_tokens": args.max_tokens, "stream": False, "format": "pcm", "laci": args.laci,
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
    parser.add_argument("--voice", default="AmericanFemale",
                        help="Server default voice (GET /v2/speakers) for a --ttfa_probe without --voice_clone.")
    parser.add_argument("--text_normalization", action=argparse.BooleanOptionalAction, default=True,
                        help="Server text normalization for the languages that have it (--no-text_normalization: off).")
    parser.add_argument("--laci", action="store_true",
                        help="LACI alignment-guarded decoding (the server needs LACI heads: serve.sh LACI_ALL=1).")
    parser.add_argument("--stream_codebooks", type=int, default=None,
                        help="TTFA probe: codebooks a frame needs before the first prefix chunk decodes it (1..n_codebooks; "
                             "n_codebooks = complete frames; default: the server's).")
    parser.add_argument("--in_process", action="store_true",
                        help="TTFA probe without the server: TTSLLM.generate_stream in a worker under the runtime's venv.")
    parser.add_argument("--runtime_dir", default="/opt/zonos3", help="--in_process: the ZONOS3 checkout (with .venv).")
    parser.add_argument("--lm", default="hf://Zyphra/ZONOS3_Mini_release_ckpts/lm/rate30_20k_r2grpo",
                        help="--in_process: the LM checkpoint (serve.sh's SERVE_MODEL).")
    parser.add_argument("--ttfa_worker", default=None, help=argparse.SUPPRESS)  # internal: the --in_process worker's texts
    args = parser.parse_args()

    print("*" * 100)
    print(f"Generating TTS with {args.model_id} on {args.dataset_path} / {args.dataset} / {args.split}")
    print("*" * 100)

    main(args)
