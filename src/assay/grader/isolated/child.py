"""Agent-code process for separate-process grading. Runs as the unprivileged agent user.

Reads one JSON request per line ({"id", "module", "func", "args", "kwargs"}), imports the
agent's module, calls the function, and writes back the result as plain JSON. Anything
the agent's code does here (exit, crash, patch things) affects only this process.
Standalone: no assay imports. The codec matches runner.py.
"""

import importlib
import json
import os
import sys


def encode(v):
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if type(v).__name__ == "Decimal" and type(v).__module__ in ("decimal", "_decimal"):
        return {"__t": "decimal", "v": str(v)}
    if isinstance(v, list):
        return [encode(x) for x in v]
    if isinstance(v, tuple):
        return {"__t": "tuple", "v": [encode(x) for x in v]}
    if isinstance(v, set):
        return {"__t": "set", "v": [encode(x) for x in v]}
    if isinstance(v, dict):
        return {"__t": "dict", "v": [[encode(k), encode(x)] for k, x in v.items()]}
    raise TypeError(f"cannot send value of type {type(v).__name__} between processes")


def decode(v):
    if isinstance(v, list):
        return [decode(x) for x in v]
    if isinstance(v, dict):
        t, items = v.get("__t"), v.get("v")
        if t == "tuple":
            return tuple(decode(x) for x in items)
        if t == "set":
            return {decode(x) for x in items}
        if t == "decimal":
            import decimal
            return decimal.Decimal(items)
        if t == "dict":
            return {decode(k): decode(x) for k, x in items}
    return v


def main() -> None:
    rfd, wfd, repo = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rf, wf = os.fdopen(rfd, "r"), os.fdopen(wfd, "w")
    sys.path.insert(0, repo)
    for line in rf:
        req = json.loads(line)
        try:
            fn = getattr(importlib.import_module(req["module"]), req["func"])
            out = {"id": req["id"], "ok": True,
                   "value": fn(*decode(req["args"]), **decode(req["kwargs"]))}
        except BaseException as e:  # noqa: BLE001 — includes SystemExit from agent code
            out = {"id": req["id"], "ok": False, "error_type": type(e).__name__,
                   "message": str(e)[:500]}
        try:
            if out["ok"]:
                out["value"] = encode(out["value"])
            text = json.dumps(out)
        except Exception as e:  # noqa: BLE001
            text = json.dumps({"id": req["id"], "ok": False, "error_type": "TypeError",
                               "message": f"return value can't be sent back: {e}"})
        wf.write(text + "\n")
        wf.flush()


if __name__ == "__main__":
    main()
