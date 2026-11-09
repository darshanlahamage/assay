"""Build notebooks/kaggle_server.ipynb from src/assay/inference/torch_server.py.

The notebook embeds the server file verbatim (tests check they match), so what runs on
Kaggle is exactly the code tested on the Mac. Re-run after editing the server:

    python tools/make_kaggle_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "src/assay/inference/torch_server.py"
OUT = ROOT / "notebooks/kaggle_server.ipynb"

INTRO = """\
# Assay remote GPU server (Kaggle 2x T4)

Runs **only the model**. The Assay harness, Docker sandbox, grader and ledger stay on your
Mac; this notebook receives conversations and returns replies, logprobs and activations.
It never executes anything the agent writes.

**Setup (once per session)**
1. Settings → Accelerator: **GPU T4 x2**. Settings → **Internet on** (needs a phone-verified account).
2. Run all cells. The last one prints two lines:
   `ASSAY_REMOTE_URL=...` and `ASSAY_REMOTE_TOKEN=...`
3. Paste both lines into `.env` in the Assay folder on the Mac (replace older ones).
   A running sweep picks them up within a minute; no restart needed.
4. Keep this tab open. Sessions end after 12 h (30 GPU-hours per week). When it ends,
   the sweep on the Mac waits; run this notebook again and paste the new lines.

**Model**: `MODEL` below. Qwen3-14B in 4-bit uses ~10 GB; Qwen3-32B in 4-bit ~20 GB
(split over both GPUs). The token is random per session: nobody else can use the URL.
"""

CONFIG = """\
MODEL = "Qwen/Qwen3-14B"     # or "Qwen/Qwen3-32B"
QUANT = "4bit"               # T4: 4bit (nf4). "none" only for small models
PORT = 8000
import os
os.environ["HF_HOME"] = "/tmp/hf"          # the model download is large; /kaggle/working holds 20 GB
"""

INSTALL = """\
!pip install -q "transformers>=4.56,<6" "bitsandbytes>=0.46" "accelerate>=1.0"
!nvidia-smi --query-gpu=name,memory.total --format=csv
"""

START = """\
import secrets, subprocess, sys, time
TOKEN = secrets.token_urlsafe(24)
log = open("/tmp/server.log", "w")
server = subprocess.Popen([sys.executable, "torch_server.py", "--model", MODEL, "--quant", QUANT,
                           "--port", str(PORT), "--token", TOKEN], stdout=log, stderr=subprocess.STDOUT)
# Download (~30 GB for 14B) and 4-bit load take ~10-15 minutes.
while server.poll() is None and "serving" not in open("/tmp/server.log").read():
    time.sleep(15)
print(open("/tmp/server.log").read()[-1500:])
assert server.poll() is None, "server exited: see the log above"
"""

SMOKE = """\
# Smoke test on the GPU: a real reply, finite numbers (fp16 can overflow), activations present.
import json, urllib.request
req = {"messages": [{"role": "system", "content": "Reply with one ```bash block."},
                    {"role": "user", "content": "List the files."}],
       "max_tokens": 300, "temperature": 0.6, "top_p": 0.95, "top_k": 20,
       "activation_layers": [0.5, 0.75], "template": {"enable_thinking": True}}
r = urllib.request.Request(f"http://127.0.0.1:{PORT}/generate", data=json.dumps(req).encode(),
                           headers={"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"})
out = json.load(urllib.request.urlopen(r, timeout=900))
print(f"{out['output_tokens']} tokens in {out['seconds']}s "
      f"({out['output_tokens'] / max(out['seconds'], 1e-6):.1f} tok/s)")
print(out["text"][-300:])
assert out["activations"] and all(x == x for x in out["entropies"]), "NaN: try QUANT='8bit'"
"""

TUNNEL = """\
# Public HTTPS URL for the Mac (Cloudflare quick tunnel; no account needed).
import re
!wget -q -O /tmp/cloudflared https://github.com/cloudflare/cloudflared/releases/download/2026.10.3/cloudflared-linux-amd64
!chmod +x /tmp/cloudflared
tlog = open("/tmp/tunnel.log", "w")
tunnel = subprocess.Popen(["/tmp/cloudflared", "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{PORT}"],
                          stdout=tlog, stderr=subprocess.STDOUT)
url = None
for _ in range(60):
    m = re.search(r"https://[a-z0-9-]+\\.trycloudflare\\.com", open("/tmp/tunnel.log").read())
    if m:
        url = m.group(0); break
    time.sleep(2)
assert url, open("/tmp/tunnel.log").read()[-2000:]
print("Paste these two lines into .env on the Mac:\\n")
print(f"ASSAY_REMOTE_URL={url}")
print(f"ASSAY_REMOTE_TOKEN={TOKEN}")
"""

KEEP = """\
# Keeps the session busy and shows progress. Stop this cell to end serving.
while server.poll() is None and tunnel.poll() is None:
    time.sleep(600)
    lines = open("/tmp/server.log").read().splitlines()
    print(time.strftime("%H:%M"), "|", lines[-1] if lines else "")
print("server or tunnel stopped; re-run the notebook")
"""


def build() -> dict:
    src = SERVER.read_text()

    def code(text: str) -> dict:
        return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": text.splitlines(keepends=True)}

    cells = [{"cell_type": "markdown", "metadata": {}, "source": INTRO.splitlines(keepends=True)},
             code(CONFIG), code(INSTALL), code("%%writefile torch_server.py\n" + src),
             code(START), code(SMOKE), code(TUNNEL), code(KEEP)]
    return {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}},
        "nbformat": 4, "nbformat_minor": 5}


def embedded_server(nb: dict) -> str:
    for c in nb["cells"]:
        s = "".join(c["source"])
        if s.startswith("%%writefile torch_server.py\n"):
            return s.split("\n", 1)[1]
    return ""


if __name__ == "__main__":
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(build(), indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
