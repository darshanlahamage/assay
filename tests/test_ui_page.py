"""The dashboard's page script runs without errors in a browser emulator (jsdom)."""

import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.mark.skipif(shutil.which("node") is None or not (TOOLS / "node_modules" / "jsdom").exists(),
                    reason="needs node and `npm install` in tools/")
def test_dashboard_page_has_no_script_errors():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = subprocess.Popen([sys.executable, "-m", "assay.cli", "ui", "--no-browser", "--port", str(port)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)
        r = subprocess.run(["node", str(TOOLS / "ui_check.js"), str(port)], capture_output=True,
                           text=True, timeout=180)
        assert r.returncode == 0, r.stdout + r.stderr
    finally:
        srv.terminate()
