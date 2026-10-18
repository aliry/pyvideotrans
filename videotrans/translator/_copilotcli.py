import subprocess
import tempfile
from dataclasses import dataclass
from types import SimpleNamespace

from videotrans.configure.config import params
from videotrans.configure.excepts import StopTask
from videotrans.translator._openaicompat import OpenAICampat

DEFAULT_COPILOTCLI_MODEL = 'claude-sonnet-5.5'


@dataclass(repr=False)
class CopilotCLI(OpenAICampat):
    """GitHub Copilot CLI channel.

    Sends each translation batch to the locally installed and logged-in
    ``copilot`` CLI in non-interactive mode instead of calling an HTTP API, so
    no API key is needed. Prompt building, SRT parsing and retries are shared
    with the other OpenAI-compatible channels.
    """

    def __post_init__(self):
        self.ainame = 'copilotcli'
        self.model_name = params.get('copilotcli_model', '') or DEFAULT_COPILOTCLI_MODEL
        # Not a URL; it only has to pass the length check in _item_task and keys the cache
        self.api_url = 'copilot-cli://local'
        self.api_key = ''
        self.reasoning_effort = None

        super().__post_init__()

    def _create_completion(self, kwargs):
        """Run one ``copilot -p`` call and wrap its reply like an OpenAI response.

        Copilot gets no tools and an empty working directory: it may only answer
        with text, never read files or run commands. A bare ``--available-tools``
        is ignored by the CLI (every tool stays enabled), so the allowlist names a
        tool that does not exist, and shell/write/url are denied as a second layer.
        """
        prompt = "\n\n".join(m['content'] for m in kwargs['messages'])
        cmd = ['copilot', '-p', prompt, '-s',
               '--model', kwargs['model'],
               '--available-tools', 'no_tools',
               '--deny-tool', 'shell', '--deny-tool', 'write', '--deny-tool', 'url',
               '--disable-builtin-mcps',
               '--disallow-temp-dir',
               '--no-color']
        try:
            with tempfile.TemporaryDirectory(prefix='copilotcli-') as empty_dir:
                proc = subprocess.run(cmd, cwd=empty_dir, capture_output=True, text=True,
                                      encoding='utf-8', timeout=kwargs.get('timeout', 300))
        except FileNotFoundError as e:
            raise StopTask('copilot CLI not found on PATH. Install GitHub Copilot CLI and run `copilot login`.') from e
        if proc.returncode != 0:
            raise RuntimeError(f'copilot CLI exited with code {proc.returncode}: {(proc.stderr or proc.stdout).strip()}')
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=proc.stdout))])
