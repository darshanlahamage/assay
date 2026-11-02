#!/usr/bin/env python3
"""apply_patch: apply a patch in the format used by OpenAI's Codex CLI.

Models trained with Codex (e.g. gpt-oss) edit files with this command, so the sandbox
provides it, like the harnesses those models were trained in.

    apply_patch <<'PATCH'
    *** Begin Patch
    *** Update File: path/to/file.py
    @@ optional context line
     unchanged line
    -removed line
    +added line
    *** Add File: new.py
    +content
    *** Delete File: old.py
    *** End Patch
    PATCH

The patch is read from the first argument if given, otherwise from stdin.
"""

from __future__ import annotations

import sys
from pathlib import Path


class PatchError(Exception):
    pass


def parse(text: str) -> list[dict]:
    lines = text.replace("\r\n", "\n").split("\n")
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip() == "*** Begin Patch")
        end = next(i for i, ln in enumerate(lines) if ln.strip() == "*** End Patch")
    except StopIteration:
        raise PatchError("patch must start with '*** Begin Patch' and end with '*** End Patch'")
    ops, cur = [], None
    for ln in lines[start + 1:end]:
        for kind in ("Add", "Update", "Delete"):
            prefix = f"*** {kind} File: "
            if ln.startswith(prefix):
                cur = {"kind": kind, "path": ln[len(prefix):].strip(), "body": [], "move": None}
                ops.append(cur)
                break
        else:
            if cur is None:
                if ln.strip():
                    raise PatchError(f"unexpected line before any file header: {ln!r}")
                continue
            if ln.startswith("*** Move to: "):
                cur["move"] = ln[len("*** Move to: "):].strip()
            elif ln.strip() == "*** End of File":
                continue
            else:
                cur["body"].append(ln)
    return ops


def hunks(body: list[str]) -> list[tuple[list[str], list[str]]]:
    out, old, new = [], [], []
    for ln in body:
        if ln.startswith("@@"):
            if old or new:
                out.append((old, new))
            old, new = [], []
            continue
        tag, text = (ln[0], ln[1:]) if ln else (" ", "")
        if tag == " ":
            old.append(text)
            new.append(text)
        elif tag == "-":
            old.append(text)
        elif tag == "+":
            new.append(text)
        else:                         # tolerate a missing leading space on context lines
            old.append(ln)
            new.append(ln)
    if old or new:
        out.append((old, new))
    return out


def find(lines: list[str], seq: list[str], start: int) -> int:
    for strip in (lambda s: s, str.rstrip, str.strip):
        target = [strip(s) for s in seq]
        for i in range(start, len(lines) - len(seq) + 1):
            if [strip(s) for s in lines[i:i + len(seq)]] == target:
                return i
    return -1


def apply(ops: list[dict], root: Path) -> list[str]:
    done = []
    for op in ops:
        path = root / op["path"]
        if op["kind"] == "Add":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(ln[1:] if ln.startswith("+") else ln
                                      for ln in op["body"]).rstrip("\n") + "\n")
            done.append(f"A {op['path']}")
        elif op["kind"] == "Delete":
            if not path.exists():
                raise PatchError(f"cannot delete missing file {op['path']}")
            path.unlink()
            done.append(f"D {op['path']}")
        else:
            if not path.exists():
                raise PatchError(f"cannot update missing file {op['path']}")
            text = path.read_text()
            lines = text.split("\n")
            pos = 0
            for old, new in hunks(op["body"]):
                if not old:                # pure addition: append at the end
                    lines[len(lines) - (lines[-1] == ""):len(lines) - (lines[-1] == "")] = new
                    continue
                i = find(lines, old, pos)
                if i < 0:
                    raise PatchError(f"{op['path']}: could not find these lines:\n" + "\n".join(old))
                lines[i:i + len(old)] = new
                pos = i + len(new)
            target = root / op["move"] if op["move"] else path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(lines))
            if op["move"]:
                path.unlink()
            done.append(f"M {op['path']}" + (f" -> {op['move']}" if op["move"] else ""))
    return done


def main() -> int:
    text = sys.argv[1] if len(sys.argv) > 1 else sys.stdin.read()
    try:
        for line in apply(parse(text), Path.cwd()):
            print(line)
    except PatchError as e:
        print(f"apply_patch: {e}", file=sys.stderr)
        return 1
    print("Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
