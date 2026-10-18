"""Copilot CLI translation channel (translate_type 30).

A fake `copilot` executable on PATH stands in for the real CLI: it records the
arguments and working directory it was started with, then prints a canned reply.
"""
import json
import os
import sys

import pytest

from videotrans.configure.config import params, settings
from videotrans.configure.excepts import StopTask
from videotrans.task.taskcfg import SrtItem
from videotrans.translator._runner import run

CANNED_REPLY = """<TRANSLATE_TEXT>
1
00:00:00,000 --> 00:00:02,000
سلام دنیا

2
00:00:02,000 --> 00:00:04,000
گاوها پول را نمی‌شناسند
</TRANSLATE_TEXT>
"""

FAKE_COPILOT = f"""#!{sys.executable}
import json, os, sys
cwd = os.getcwd()
with open(os.environ['FAKE_COPILOT_LOG'], 'w', encoding='utf-8') as f:
    json.dump({{'argv': sys.argv[1:], 'cwd': cwd, 'cwd_files': os.listdir(cwd)}}, f)
sys.stderr.write(os.environ.get('FAKE_COPILOT_STDERR', ''))
with open(os.environ['FAKE_COPILOT_STDOUT'], encoding='utf-8') as f:
    sys.stdout.write(f.read())
sys.exit(int(os.environ.get('FAKE_COPILOT_EXIT', '0')))
"""


def _srt_items():
    return [
        SrtItem(text="Hello world", line=1, start_time=0, end_time=2000,
                startraw="00:00:00,000", endraw="00:00:02,000",
                time="00:00:00,000 --> 00:00:02,000"),
        SrtItem(text="Cows don't know money", line=2, start_time=2000, end_time=4000,
                startraw="00:00:02,000", endraw="00:00:04,000",
                time="00:00:02,000 --> 00:00:04,000"),
    ]


class FakeCopilot:
    def __init__(self, tmp_path, monkeypatch):
        self.log = tmp_path / "call.json"
        self.stdout = tmp_path / "stdout.txt"
        self.monkeypatch = monkeypatch
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        exe = bin_dir / "copilot"
        exe.write_text(FAKE_COPILOT, encoding="utf-8")
        exe.chmod(0o755)
        monkeypatch.setenv("PATH", str(bin_dir))
        monkeypatch.setenv("FAKE_COPILOT_LOG", str(self.log))
        monkeypatch.setenv("FAKE_COPILOT_STDOUT", str(self.stdout))
        self.reply(CANNED_REPLY)

    def reply(self, stdout, exit_code=0, stderr=""):
        self.stdout.write_text(stdout, encoding="utf-8")
        self.monkeypatch.setenv("FAKE_COPILOT_EXIT", str(exit_code))
        self.monkeypatch.setenv("FAKE_COPILOT_STDERR", stderr)

    def call(self):
        return json.loads(self.log.read_text(encoding="utf-8"))


@pytest.fixture
def fake_copilot(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "aisendsrt", True, raising=False)
    monkeypatch.setattr(params, "copilotcli_model", "", raising=False)
    return FakeCopilot(tmp_path, monkeypatch)


def _translate():
    return run(translate_type=30, text_list=_srt_items(),
               source_code="en", target_code="fa", is_test=True)


def test_translates_srt_batch_through_copilot_cli(fake_copilot):
    result = _translate()

    assert [it["text"] for it in result] == ["سلام دنیا", "گاوها پول را نمی‌شناسند"]
    argv = fake_copilot.call()["argv"]
    assert argv[argv.index("--model") + 1] == "claude-sonnet-5.5"
    assert "-s" in argv
    prompt = argv[argv.index("-p") + 1]
    # SRT mode: numbered blocks with their timestamps, target language by name
    assert "2\n00:00:02,000 --> 00:00:04,000\nCows don't know money" in prompt
    assert "Persian" in prompt


# Tool names copilot 1.0.98 exposes by default (listed by the real CLI).
REAL_COPILOT_TOOLS = {
    "bash", "read_bash", "stop_bash", "list_bash", "view", "create", "edit",
    "web_fetch", "fetch_copilot_cli_documentation", "skill", "run_dynamic_workflow",
    "dynamic_workflows_manage", "sql", "session_store_sql", "read_agent",
    "list_agents", "write_agent", "grep", "glob", "task",
}


def test_copilot_gets_no_tools_and_an_empty_working_dir(fake_copilot):
    _translate()

    call = fake_copilot.call()
    argv = call["argv"]
    # A bare `--available-tools` is ignored by copilot and leaves every tool enabled;
    # the allowlist must name something, and nothing real.
    allowed = argv[argv.index("--available-tools") + 1]
    assert not allowed.startswith("-")
    assert not set(allowed.split(",")) & REAL_COPILOT_TOOLS
    denied = {argv[i + 1] for i, a in enumerate(argv) if a == "--deny-tool"}
    assert {"shell", "write", "url"} <= denied
    assert "--disable-builtin-mcps" in argv
    assert call["cwd_files"] == []
    assert os.path.realpath(call["cwd"]) != os.path.realpath(os.getcwd())


def test_model_comes_from_copilotcli_model_param(fake_copilot, monkeypatch):
    monkeypatch.setattr(params, "copilotcli_model", "gpt-5.5", raising=False)

    _translate()

    argv = fake_copilot.call()["argv"]
    assert argv[argv.index("--model") + 1] == "gpt-5.5"


def test_missing_copilot_binary_stops_task(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "aisendsrt", True, raising=False)
    empty = tmp_path / "empty"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))

    with pytest.raises(StopTask, match="copilot"):
        _translate()


def test_failed_copilot_call_reports_its_stderr(fake_copilot):
    fake_copilot.reply("", exit_code=1, stderr="Error: not authenticated, run copilot login")

    with pytest.raises(Exception, match="not authenticated"):
        _translate()
