"""GPU model server for the remote track (Kaggle / Colab), the PyTorch twin of engine.py.

The harness, Docker sandbox, grader and ledger stay on the Mac. This server only turns a
conversation into the next reply, and returns what the MLX engine returns locally:

  text            the raw generation (the Mac parses commands and <think> reasoning out of it)
  logprobs        chosen token's log-probability, per generated token
  entropies       entropy (nats) of the full next-token distribution, per generated token
  activations     residual stream at the requested relative depths, at three positions:
                    prompt_last  last prompt token        action  first token of the action block
                    gen_mean     mean over generated tokens
                  base64 .npz, float16, each [n_layers, hidden]

Nothing the agent writes is executed here: requests carry text, replies carry text and
numbers. Requests need the bearer token printed at start-up.

Self-contained on purpose (stdlib + torch + transformers + numpy, imported lazily), so
the Kaggle notebook can run this one file without installing Assay, which needs MLX.

    python torch_server.py --model Qwen/Qwen3-14B --quant 4bit --port 8000
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import secrets
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

ACTION_OPENERS = ("```bash\n", "```submit\n")


def action_token(pieces: list[str]) -> int:
    """Index of the first generated token inside the action block (or 0). Same as engine.py."""
    text = ""
    for i, p in enumerate(pieces):
        text += p
        for opener in ACTION_OPENERS:
            j = text.find(opener)
            if j != -1 and len(text) > j + len(opener):
                return i
    return 0


def layer_indices(depths: list[float], n_layers: int) -> list[int]:
    """Relative depths (0.5) or absolute indices (12) -> layer indices. Same rule as engine.py."""
    return [int(round(x * (n_layers - 1))) if isinstance(x, float) else int(x) for x in depths]


def collect(records: list, layers: list[int], n_prompt: int, pieces: list[str]) -> dict:
    """Per-call records (layer, tokens in call, last hidden) -> three [n_layers, hidden] arrays.

    Mirrors engine._collect: the call that ends on the last prompt token gives prompt_last;
    each later call processed one generated token."""
    import numpy as np

    n_gen = len(pieces)
    at = action_token(pieces)
    out: dict[str, list] = {"prompt_last": [], "action": [], "gen_mean": []}
    for layer in layers:
        seen, last, gen = 0, None, []
        for li, n, h in records:
            if li != layer:
                continue
            seen += n
            if seen == n_prompt:
                last = h
            elif seen > n_prompt and len(gen) < n_gen:
                gen.append(h)
        d = (last if last is not None else gen[0] if gen else np.zeros(1)).shape[-1]
        out["prompt_last"].append(last if last is not None else np.zeros(d, np.float32))
        if gen:
            g = np.stack(gen)
            out["gen_mean"].append(g.mean(0))
            out["action"].append(g[min(at, len(g) - 1)])
        else:
            out["gen_mean"].append(np.zeros(d, np.float32))
            out["action"].append(np.zeros(d, np.float32))
    return {k: np.stack(v).astype(np.float16) for k, v in out.items()}


def pack(arrays: dict) -> str:
    import numpy as np

    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return base64.b64encode(buf.getvalue()).decode()


class Engine:
    def __init__(self, model: str, quant: str = "4bit", dtype: str = "float16",
                 device_map: str = "auto") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.torch = torch
        self.name = model
        # fp16: T4s have no bf16. float32 for CPU smoke tests.
        # "single": the whole model on GPU 0 (no hop between cards per token) when it fits.
        dm: Any = {"": 0} if device_map == "single" and torch.cuda.is_available() else "auto"
        kw: dict[str, Any] = {"device_map": dm, "dtype": getattr(torch, dtype)}
        if quant == "4bit":
            kw["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
        elif quant == "8bit":
            kw["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model, **kw).eval()
        self.layers = self.model.model.layers
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos]) | {self.tok.eos_token_id}
        self.records: list = []
        self.hooked: list[int] = []
        self._handles: list = []
        self.cache, self.cache_tokens = None, []
        self.lock = threading.Lock()

    def _hook(self, layers: list[int]) -> None:
        if layers == self.hooked:
            return
        for h in self._handles:
            h.remove()
        self._handles = []
        for i in layers:
            def fn(_m, _inp, out, i=i):
                h = out[0] if isinstance(out, tuple) else out
                self.records.append((i, h.shape[1], h[0, -1].detach().float()))   # stays on GPU
            self._handles.append(self.layers[i].register_forward_hook(fn))
        self.hooked = layers

    def generate(self, req: dict) -> dict:
        torch = self.torch
        from transformers import DynamicCache

        prompt = self.tok.apply_chat_template(
            req["messages"], tools=req.get("tools"), tokenize=False, add_generation_prompt=True,
            **(req.get("template") or {}))
        tokens = self.tok.encode(prompt, add_special_tokens=False)
        max_tokens = int(req.get("max_tokens", 4096))
        temp = float(req.get("temperature", 0.7))
        top_p, top_k = float(req.get("top_p", 0.0)), int(req.get("top_k", 0))
        presence = float(req.get("presence_penalty", 0.0))
        layers = layer_indices(req.get("activation_layers") or [], len(self.layers))
        seed = req.get("seed")

        with self.lock, torch.inference_mode():
            self._hook(layers)
            # Reuse the KV cache for the longest shared prefix; always feed one token.
            common = 0
            if self.cache is not None:
                limit = min(len(tokens) - 1, len(self.cache_tokens))
                while common < limit and tokens[common] == self.cache_tokens[common]:
                    common += 1
                if common == 0:
                    self.cache = DynamicCache()
                else:
                    # crop(-n) removes n tokens (transformers 4 and 5 both accept it)
                    extra = self.cache.get_seq_length() - common
                    if extra > 0:
                        self.cache.crop(-extra)
            else:
                self.cache = DynamicCache()
            suffix = tokens[common:]
            self.records.clear()
            device = self.model.get_input_embeddings().weight.device
            ids = torch.tensor([suffix], device=device)
            out_ids: list[int] = []
            lps, ents = [], []                       # GPU scalars; copied once at the end
            vocab = self.model.config.vocab_size
            out_dev = self.model.get_output_embeddings().weight.device
            seen = torch.zeros(vocab, dtype=torch.bool, device=out_dev)
            gen_ = torch.Generator(device=out_dev)
            gen_.manual_seed(int(seed) if seed is not None else int(time.time_ns() % 2**31))
            try:
                for _ in range(max_tokens):
                    # Sampling stays on the GPU: only the chosen token id comes back per step.
                    logits = self.model(input_ids=ids, past_key_values=self.cache,
                                        use_cache=True).logits[0, -1, :vocab].float()
                    if presence and out_ids:
                        logits = logits - presence * seen.float()
                    lp = torch.log_softmax(logits, -1)
                    if temp <= 0:
                        t = int(lp.argmax())
                    else:
                        z = logits / temp
                        if top_k > 0:
                            kth = torch.topk(z, top_k).values[-1]
                            z = z.masked_fill(z < kth, float("-inf"))
                        p = torch.softmax(z, -1)
                        if 0 < top_p < 1:
                            sp, si = torch.sort(p, descending=True)
                            drop = sp.cumsum(0) - sp > top_p
                            sp[drop] = 0
                            p = torch.zeros_like(p).scatter(0, si, sp)
                            p = p / p.sum()
                        t = int(torch.multinomial(p, 1, generator=gen_))
                    lps.append(lp[t])
                    ents.append(-(lp.exp() * lp).nan_to_num().sum())
                    out_ids.append(t)
                    seen[t] = True
                    if t in self.eos:
                        break
                    ids = torch.tensor([[t]], device=device)
                logprobs = torch.stack(lps).cpu().tolist() if lps else []
                entropies = torch.stack(ents).cpu().tolist() if ents else []
                if not all(x == x and abs(x) != float("inf") for x in logprobs):
                    raise FloatingPointError("non-finite logits (fp16 overflow?)")
            except BaseException:
                self.cache, self.cache_tokens = None, []
                raise
            self.records[:] = [(i, n, h.cpu().numpy()) for i, n, h in self.records]
            held = tokens + out_ids
            n_cached = self.cache.get_seq_length()
            self.cache_tokens = held[:n_cached]
            body = [t for t in out_ids if t not in self.eos]
            pieces = [self.tok.decode([t]) for t in body]
            acts = collect(self.records, layers, len(suffix), pieces) if layers else {}
            self.records.clear()
        return {"text": self.tok.decode(body), "prompt_tokens": len(tokens),
                "cached_tokens": common, "output_tokens": len(out_ids),
                "finish_reason": "stop" if out_ids and out_ids[-1] in self.eos else "length",
                "logprobs": logprobs, "entropies": entropies,
                "activations": pack(acts) if acts else None}


def serve(engine: Any, token: str, port: int) -> ThreadingHTTPServer:
    jobs: dict[str, dict] = {}
    jobs_lock = threading.Lock()

    def run(req: dict) -> dict:
        try:
            t0 = time.time()
            res = engine.generate(req)
            res["seconds"] = round(time.time() - t0, 1)
            print(f"[{time.strftime('%H:%M:%S')}] prompt {res['prompt_tokens']} "
                  f"(cached {res['cached_tokens']}) -> {res['output_tokens']} tokens "
                  f"in {res['seconds']}s", flush=True)
            return {"status": "done", **res}
        except Exception as e:  # noqa: BLE001 — report to the client, keep serving
            print(f"error: {type(e).__name__}: {e}", flush=True)
            return {"status": "error", "error": f"{type(e).__name__}: {e}"}

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authed(self) -> bool:
            if secrets.compare_digest(self.headers.get("Authorization", ""), f"Bearer {token}"):
                return True
            self._send(401, {"error": "bad token"})
            return False

        def do_GET(self) -> None:  # noqa: N802
            if not self._authed():
                return
            if self.path == "/health":
                return self._send(200, {"model": engine.name, "layers": len(engine.layers)})
            if self.path.startswith("/jobs/"):
                job = jobs.get(self.path[len("/jobs/"):])
                return self._send(200, job) if job else self._send(404, {"error": "unknown job"})
            self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if self.path not in ("/generate", "/jobs"):
                return self._send(404, {"error": "not found"})
            if not self._authed():
                return
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/generate":                # synchronous: local smoke tests only
                res = run(req)
                return self._send(200 if res["status"] == "done" else 500, res)
            # Asynchronous: tunnels cut requests off at ~100 s, a reply can take minutes.
            # The client names the job, so re-submitting after a lost response is harmless.
            jid = str(req.get("id") or secrets.token_hex(8))
            with jobs_lock:
                if jid not in jobs:
                    jobs[jid] = {"status": "running", "id": jid}
                    for old in list(jobs)[:-50]:        # keep the last 50
                        jobs.pop(old, None)
                    threading.Thread(target=lambda: jobs.__setitem__(jid, {**run(req), "id": jid}),
                                     daemon=True).start()
            self._send(200, {"id": jid, "status": jobs[jid]["status"]})

        def log_message(self, *a: Any) -> None:
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-14B")
    ap.add_argument("--quant", default="4bit", choices=["4bit", "8bit", "none"])
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--token", default=None)
    ap.add_argument("--dtype", default="float16", choices=["float16", "float32", "bfloat16"])
    ap.add_argument("--device-map", default="auto", choices=["auto", "single"])
    a = ap.parse_args()
    token = a.token or secrets.token_urlsafe(24)
    engine = Engine(a.model, a.quant, a.dtype, a.device_map)
    serve(engine, token, a.port)
    print(f"serving {a.model} on 127.0.0.1:{a.port}\nASSAY_REMOTE_TOKEN={token}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
