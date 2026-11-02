"""Lab 8. Inference and serving: cache, INT8, speculative decoding, an
OpenAI-compatible API (Chapters 15-18).

Steps:
  1. generation with and without the KV/latent cache: identical tokens,
     time per token as the text grows (prefill vs decode, Section 15.1);
  2. weight-only INT8 (formula (15.4)): memory and quality (perplexity,
     arithmetic accuracy);
  3. self-speculative decoding with the MTP module: acceptance rate, tokens
     per main-model step (formula (7.11)), equality with greedy decoding;
  4. serve the model with tinyseek.serve, send chat requests over HTTP like an
     OpenAI client, measure latency and throughput and estimate the cost of
     one million output tokens on your own server (formula (16.3)).

Run (from code/):
    python tinyseek_labs/lab8_serve.py
    python tinyseek_labs/lab8_serve.py --price-per-hour 0.5 --utilization 0.3
Outputs: runs/lab8/cache.csv, int8.csv, speculative.csv, serving.csv, cache.svg
"""
import argparse
import json
import threading
import time
import urllib.request

import numpy as np
from _common import (RUNS, Timer, base_model, print_table, run_dir, sft_model, svg_plot,
                     write_csv)
import torch

from tinyseek.data import PackedDataset, make_arith_task, prepare_data
from tinyseek.evaluate import arith_accuracy, perplexity
from tinyseek.generate import generate_ids, quantize_int8, speculative_generate, weight_bytes
from tinyseek.model import cache_tensor_bytes, kv_cache_bytes
from tinyseek.serve import make_server
from tinyseek.train import load_checkpoint


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--new-tokens", type=int, default=160, help="tokens for timing")
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--price-per-hour", type=float, default=0.10,
                    help="rent of the server, price units per hour")
    ap.add_argument("--utilization", type=float, default=0.5, help="u in formula (16.3)")
    ap.add_argument("--temperature", type=float, default=1.0, help="for sampled speculation")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    out, timer = run_dir(8), Timer()
    base, tok, base_dir = base_model()
    sft, _, sft_dir = sft_model()
    _, data = prepare_data(RUNS / "data", 2048, args.seed, verbose=False)
    torch.manual_seed(args.seed)

    # 1. cache --------------------------------------------------------------------
    prompt = torch.as_tensor(data["val"][:64].astype("int64"))[None]
    rows, curves = [], {}
    for use_cache in (True, False):
        t0 = time.perf_counter()
        new, logits = generate_ids(base, prompt, args.new_tokens, temperature=0,
                                   use_cache=use_cache, return_logits=True)
        dt = time.perf_counter() - t0
        rows.append({"mode": "cache" if use_cache else "no cache", "tokens": new.shape[1],
                     "ms_per_token": round(dt / new.shape[1] * 1e3, 2),
                     "tokens_per_s": round(new.shape[1] / dt, 1)})
        if use_cache:
            ref_new, ref_logits = new, logits
    same = torch.equal(ref_new, new)
    diff = max(float((a - b).abs().max()) for a, b in zip(ref_logits, logits))
    print_table(rows, f"Greedy generation of {args.new_tokens} tokens after a 64-token prompt")
    print(f"identical tokens: {same}; max |logit difference| = {diff:.2e}")
    steps_t = {"cache": [], "no cache": []}
    lens = list(range(16, 241, 32))
    for L in lens:
        ids = torch.as_tensor(data["val"][:L].astype("int64"))[None]
        with torch.no_grad():
            c = base.new_cache()
            base(ids, cache=c)
            t0 = time.perf_counter()
            for _ in range(5):
                base(ids[:, -1:], cache=c)
            steps_t["cache"].append((time.perf_counter() - t0) / 5 * 1e3)
            last_cache = c
            t0 = time.perf_counter()
            for _ in range(5):
                base(ids)
            steps_t["no cache"].append((time.perf_counter() - t0) / 5 * 1e3)
    svg_plot(out / "cache.svg", {k: (lens, v) for k, v in steps_t.items()},
             "Time of one generated token vs context length", "context (tokens)", "ms")
    crow = [{"context": L, "ms_cache": round(a, 2), "ms_no_cache": round(b, 2),
             "cache_KB_fp32": round(kv_cache_bytes(base.config, L, bytes_per_elem=4) / 1024, 1)}
            for L, a, b in zip(lens, steps_t["cache"], steps_t["no cache"])]
    print_table(crow, "One decoding step: cached vs recomputed")
    print(f"cache tensors after the last run: {cache_tensor_bytes(last_cache) / 1024:.1f} KB")
    write_csv(out / "cache.csv", rows + crow)

    # 2. INT8 -----------------------------------------------------------------------
    irows = []
    va = PackedDataset(data["val"], 256)
    for name, m in (("fp32", base), ("int8", quantize_int8(load_checkpoint(base_dir)[0]))):
        irows.append({"model": f"base {name}", "MB": round(weight_bytes(m) / 1e6, 2),
                      "val_ppl": round(perplexity(m, va, max_batches=16), 3)})
    for name, m in (("fp32", sft), ("int8", quantize_int8(load_checkpoint(sft_dir)[0]))):
        acc = arith_accuracy(m, tok, n=200, digits=(1, 2), seed=4242)
        irows.append({"model": f"sft {name}", "MB": round(weight_bytes(m) / 1e6, 2),
                      "arith_accuracy": round(acc["accuracy"], 3)})
    print_table(irows, "Weight-only INT8, one scale per output channel")
    write_csv(out / "int8.csv", irows)

    # 3. speculative decoding --------------------------------------------------------
    rng = np.random.default_rng(args.seed)
    prose = [data["val_by_source"][s][i * 97:i * 97 + 32].tolist()
             for s in ("uk", "en", "code") if s in data["val_by_source"] for i in range(3)]
    arith = [tok.encode(make_arith_task(rng, 2)[0]) for _ in range(9)]
    srows = []
    for label, model, prompts in (("base, text", base, prose), ("sft, arithmetic", sft, arith)):
        for temp in (0.0, args.temperature):
            acc, tps, equal, t_spec, t_plain = [], [], [], 0.0, 0.0
            g = torch.Generator().manual_seed(args.seed)
            for p in prompts:
                t0 = time.perf_counter()
                new, st = speculative_generate(model, p, 48, temp, generator=g,
                                               eos_id=tok.eos_id)
                t_spec += time.perf_counter() - t0
                acc.append(st["acceptance"])
                tps.append(st["tokens_per_step"])
                if temp == 0:
                    t0 = time.perf_counter()
                    ref = generate_ids(model, torch.tensor([p]), 48, 0,
                                       eos_id=tok.eos_id)[0].tolist()
                    t_plain += time.perf_counter() - t0
                    equal.append(ref[:len(new)] == new)
            srows.append({"model/prompts": label, "temperature": temp,
                          "acceptance": round(float(np.mean(acc)), 3),
                          "tokens_per_step": round(float(np.mean(tps)), 3),
                          "same_as_greedy": f"{sum(equal)}/{len(equal)}" if equal else "-",
                          "wall_speedup": round(t_plain / t_spec, 2) if equal else "-"})
    print_table(srows, "Self-speculative decoding with the MTP module (48 new tokens)")
    write_csv(out / "speculative.csv", srows)

    # 4. OpenAI-compatible server ------------------------------------------------------
    server = make_server(sft, tok, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1/chat/completions"
    lat, n_out, correct = [], 0, 0
    t_all = time.perf_counter()
    for i in range(args.requests):
        q, a = make_arith_task(rng, 2)
        question = q.split("Q: ")[1].split("\n")[0]
        body = json.dumps({"model": "tinyseek", "temperature": 0, "max_tokens": 48,
                           "messages": [{"role": "user", "content": question}]}).encode()
        t0 = time.perf_counter()
        req = urllib.request.Request(url, body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req) as r:
            resp = json.load(r)
        lat.append(time.perf_counter() - t0)
        n_out += resp["usage"]["completion_tokens"]
        correct += f"<answer>{a}</answer>" in resp["choices"][0]["message"]["content"]
        if i == 0:
            print(f"\nrequest : {question}\nresponse: {resp['choices'][0]['message']['content']}"
                  f"\nusage   : {resp['usage']}")
    wall = time.perf_counter() - t_all
    server.shutdown()
    server.server_close()
    tput = n_out / wall
    cost = args.price_per_hour / (3600 * tput * args.utilization) * 1e6
    sv = [{"requests": args.requests, "p50_latency_s": round(float(np.median(lat)), 3),
           "p95_latency_s": round(float(np.percentile(lat, 95)), 3),
           "output_tokens_per_s": round(tput, 1), "answers_correct": f"{correct}/{args.requests}",
           "price_per_hour": args.price_per_hour, "utilization": args.utilization,
           "cost_per_1M_output_tokens": round(cost, 3)}]
    print_table(sv, "tinyseek.serve: sequential chat requests (formula (16.3) for the cost)")
    write_csv(out / "serving.csv", sv)
    print(f"\nsaved to {out}/ in {timer}")


if __name__ == "__main__":
    main()
