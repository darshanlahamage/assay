from assay.agent import protocol


def test_parse_bash():
    a = protocol.parse("Look at the files.\n```bash\nls -la\n```\n")
    assert (a.kind, a.content) == ("bash", "ls -la")
    assert a.reasoning == "Look at the files."


def test_parse_submit():
    a = protocol.parse("Done.\n```submit\nFixed get().\n```")
    assert (a.kind, a.content) == ("submit", "Fixed get().")


def test_two_blocks_runs_the_first():
    a = protocol.parse("```bash\nls\n```\n```bash\npwd\n```")
    assert (a.kind, a.content, a.extra_blocks) == ("bash", "ls", 1)


def test_escaped_newlines_are_restored():
    a = protocol.parse("```bash\ncat > f <<'EOF'\\nline1\\nline2\\nEOF\n```")
    assert a.content == "cat > f <<'EOF'\nline1\nline2\nEOF"


def test_no_block_is_format_error():
    assert protocol.parse("I will now fix it.").kind == "format_error"


def test_blocks_inside_think_are_ignored():
    a = protocol.parse("<think>maybe ```bash\nrm -rf /\n```</think>\n```bash\nls\n```")
    assert (a.kind, a.content) == ("bash", "ls")


def test_observation_truncates_middle():
    obs = protocol.format_observation(0, "a" * 50 + "b" * 50, False, limit=20)
    assert obs.startswith("exit code: 0\naaaaaaaaaa") and obs.endswith("bbbbbbbbbb")
    assert "characters omitted" in obs
