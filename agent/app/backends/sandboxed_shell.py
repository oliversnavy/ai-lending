"""
SandboxedLocalShellBackend — LocalShellBackend with execute() run inside an
ephemeral Docker container instead of directly on the host.

Background: on 2026-07-11, T1a episode_0002 hit a pickling error (the
sensitivity model pickles reference `data_pipeline.sensitivity_model`, which
wasn't importable from the episode's own subprocess) and "fixed" it by
running `cat > /home/.../data_pipeline/sensitivity_model.py << EOF` via the
execute() tool — overwriting the real, shared project module with an 11-line
stub. HarnessGuardrailsMiddleware only pattern-matches a blocklist of
dangerous commands (rm -rf, curl, sudo, ...); it never checked whether a
shell command's output redirection targeted a path outside skill_dir, so an
absolute-path `cat > ... << EOF` sailed straight through. That's a structural
gap in LocalShellBackend itself, not just a missing regex: the upstream
library documents that `virtual_mode` and path-based guardrails provide "NO
security with shell access enabled, since commands can access any path on
the system."

Only execute() is overridden here. write_file/read_file/edit_file/ls/glob/
grep are inherited from FilesystemBackend (direct host filesystem I/O via
Python, not shell) and are unchanged — those are already correctly gated by
HarnessGuardrailsMiddleware._check_write_path(), which restricts them to
skill_dir. The gap was specifically shell commands bypassing that check
entirely, so this only needed to sandbox execute().

Each execute() call runs in a fresh, ephemeral `docker run --rm` container:
  - project_root, /usr, /etc bind-mounted read-only — same read access the
    agent already had (this does not restrict what the agent can read, only
    what it can write)
  - skill_dir bind-mounted read-write on top of the read-only project_root
    mount — the only writable path inside the sandbox
  - --network none — structurally blocks curl/wget/exfiltration at the
    kernel level, regardless of the regex blocklist
  - PATH is set explicitly with {project_root}/.venv/bin first, so bare
    `python3` in agent-authored commands resolves to the project's venv
    (matching existing behavior when the harness itself runs under `uv run`)

Any write outside skill_dir now fails at the kernel/mount level ("Read-only
file system"), not via pattern matching — this closes the class of bug, not
just the one command that triggered it.

Chosen over bubblewrap: bwrap's unprivileged sandboxing is blocked on this
host by Ubuntu's apparmor_restrict_unprivileged_userns policy (no bwrap
AppArmor profile installed); enabling it needs a system-wide sudo policy
change. Docker requires no privilege change (the harness user is already in
the `docker` group) at the cost of a per-command container-startup latency
(~1-2s), which is negligible against the existing 180s execute() timeout.
"""
from __future__ import annotations

import pathlib
import subprocess
import uuid

from deepagents.backends.local_shell import LocalShellBackend
from deepagents.backends.protocol import ExecuteResponse

_SANDBOX_IMAGE = "ubuntu:24.04"
_STARTUP_OVERHEAD_SECONDS = 15  # docker run cold-start budget, added on top of the command's own timeout


class SandboxedLocalShellBackend(LocalShellBackend):
    """LocalShellBackend whose execute() runs inside a locked-down, ephemeral Docker container.

    Accepts the same constructor arguments as LocalShellBackend, plus one
    required keyword: project_root, the read-only mount covering everything
    the agent should still be able to read (data, source, venv) but never
    write outside of root_dir (skill_dir).
    """

    def __init__(self, *args, project_root: pathlib.Path, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._project_root = pathlib.Path(project_root).resolve()
        self._venv_bin = self._project_root / ".venv" / "bin"

    def execute(self, command: str, *, timeout: int | None = None) -> ExecuteResponse:
        if not command or not isinstance(command, str):
            return ExecuteResponse(output="Error: Command must be a non-empty string.", exit_code=1, truncated=False)

        effective_timeout = timeout if timeout is not None else self._default_timeout
        if effective_timeout <= 0:
            msg = f"timeout must be positive, got {effective_timeout}"
            raise ValueError(msg)

        skill_dir = str(self.cwd)
        container_name = f"sandbox-{uuid.uuid4().hex[:12]}"
        container_path = (
            f"{self._venv_bin}:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
        )
        docker_cmd = [
            "docker", "run", "--rm",
            "--name", container_name,
            "--network", "none",
            "--memory", "16g",
            "-v", f"{self._project_root}:{self._project_root}:ro",
            "-v", "/usr:/usr:ro",
            "-v", "/etc:/etc:ro",
            "-v", f"{skill_dir}:{skill_dir}:rw",
            "-w", skill_dir,
            "-e", f"PATH={container_path}",
            _SANDBOX_IMAGE,
            "/bin/sh", "-c", command,
        ]

        try:
            result = subprocess.run(  # noqa: S603
                docker_cmd,
                check=False,
                capture_output=True,
                stdin=subprocess.DEVNULL,
                text=True,
                timeout=effective_timeout + _STARTUP_OVERHEAD_SECONDS,
                env=self._env,
            )
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", container_name], capture_output=True)  # noqa: S603, S607
            msg = (
                f"Error: Command timed out after {effective_timeout} seconds. "
                "For long-running commands, re-run using the timeout parameter."
            )
            return ExecuteResponse(output=msg, exit_code=124, truncated=False)
        except Exception as e:  # noqa: BLE001
            return ExecuteResponse(
                output=f"Error executing command ({type(e).__name__}): {e}",
                exit_code=1,
                truncated=False,
            )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            stderr_lines = result.stderr.strip().split("\n")
            output_parts.extend(f"[stderr] {line}" for line in stderr_lines)
        output = "\n".join(output_parts) if output_parts else "<no output>"

        truncated = False
        if len(output) > self._max_output_bytes:
            output = output[: self._max_output_bytes]
            output += f"\n\n... Output truncated at {self._max_output_bytes} bytes."
            truncated = True

        if result.returncode != 0:
            output = f"{output.rstrip()}\n\nExit code: {result.returncode}"

        return ExecuteResponse(output=output, exit_code=result.returncode, truncated=truncated)


__all__ = ["SandboxedLocalShellBackend"]
