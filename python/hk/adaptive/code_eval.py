"""
HK Adaptive Neural Framework: Persistent Code Evaluation Sandbox
Provides sandboxed code execution, test verification, syntax checking,
and persistent serialization of evaluation histories into HK containers.
"""

import os
import sys
import ast
import json
import time
import shutil
import tempfile
import subprocess
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple

@dataclass
class TestCase:
    __test__ = False
    input_call: str
    expected_output: Any
    description: str = ""

@dataclass
class EvalResult:
    success: bool
    pass_rate: float
    total_tests: int
    passed_tests: int
    execution_time_ms: float
    stdout: str = ""
    stderr: str = ""
    syntax_valid: bool = True
    error_message: str = ""
    test_details: List[Dict[str, Any]] = None

    def to_json(self) -> str:
        d = asdict(self)
        return json.dumps(d)

    @classmethod
    def from_json(cls, json_str: str) -> "EvalResult":
        d = json.loads(json_str)
        return cls(**d)

class CodeSandbox:
    """
    Isolated execution sandbox with Docker container isolation, timeouts, and resource ceilings.
    When Docker is available, executes arbitrary model-generated Python code inside an ephemeral,
    unprivileged container with network isolation (--network none) and strict memory/CPU caps.
    """
    def __init__(
        self,
        timeout_sec: float = 3.0,
        python_executable: Optional[str] = None,
        use_docker: bool = False,
        docker_image: str = "python:3.10-slim",
        memory_limit: str = "256m",
        cpu_limit: float = 1.0,
    ):
        self.timeout_sec = timeout_sec
        self.python_exe = python_executable or sys.executable
        self.use_docker = use_docker
        self.docker_image = docker_image
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self._docker_available = self._detect_docker() if self.use_docker else False

    def _detect_docker(self) -> bool:
        docker_path = shutil.which("docker")
        if not docker_path:
            return False
        try:
            res = subprocess.run([docker_path, "info"], capture_output=True, timeout=2.0)
            if res.returncode != 0:
                return False
            inspect_res = subprocess.run(
                [docker_path, "image", "inspect", self.docker_image],
                capture_output=True,
                timeout=2.0
            )
            return inspect_res.returncode == 0
        except Exception:
            return False

    @property
    def is_docker_active(self) -> bool:
        return self.use_docker and self._docker_available

    def check_syntax(self, code_str: str) -> Tuple[bool, str]:
        """Validates Python syntax using AST parser."""
        try:
            ast.parse(code_str)
            return True, ""
        except SyntaxError as e:
            return False, f"SyntaxError at line {e.lineno}: {e.msg}"

    def execute_code(
        self,
        code_str: str,
        test_cases: Optional[List[TestCase]] = None
    ) -> EvalResult:
        """
        Executes code against optional unit test cases in a sandboxed child process or Docker container.
        Measures elapsed time, captures stdout/stderr, and returns structured results.
        """
        # 1. Syntax check
        valid_syntax, syn_err = self.check_syntax(code_str)
        if not valid_syntax:
            return EvalResult(
                success=False,
                pass_rate=0.0,
                total_tests=len(test_cases) if test_cases else 0,
                passed_tests=0,
                execution_time_ms=0.0,
                syntax_valid=False,
                error_message=syn_err,
                test_details=[]
            )

        # 2. Build execution harness script
        harness_code = [code_str, "\n\n# --- Test Execution Harness ---"]
        if test_cases:
            harness_code.append("import json, sys")
            harness_code.append("results = []")
            for i, tc in enumerate(test_cases):
                harness_code.append(f"""
try:
    _out = {tc.input_call}
    _expected = {repr(tc.expected_output)}
    _passed = (_out == _expected)
    results.append({{"test": {i}, "passed": _passed, "actual": str(_out), "expected": str(_expected)}})
except Exception as _e:
    results.append({{"test": {i}, "passed": False, "error": str(_e)}})
""")
            harness_code.append("print('__TEST_RESULTS_JSON__' + json.dumps(results))")

        full_script = "\n".join(harness_code)

        # 3. Execute in Docker container or fallback subprocess
        start_t = time.perf_counter()
        is_running_in_docker = self.use_docker and self._docker_available

        proc = None
        if is_running_in_docker:
            cmd = [
                "docker", "run", "--rm", "-i",
                "--network", "none",
                "--memory", self.memory_limit,
                f"--cpus={self.cpu_limit}",
                self.docker_image,
                "python", "-"
            ]
            try:
                proc = subprocess.run(
                    cmd,
                    input=full_script,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec,
                )
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                stdout = proc.stdout
                stderr = proc.stderr
                return_code = proc.returncode
                # Fallback to subprocess if Docker infrastructure error occurred (exit 125)
                if return_code == 125 or "no matching manifest" in (stderr or ""):
                    proc = None
            except subprocess.TimeoutExpired:
                elapsed_ms = self.timeout_sec * 1000.0
                return EvalResult(
                    success=False,
                    pass_rate=0.0,
                    total_tests=len(test_cases) if test_cases else 0,
                    passed_tests=0,
                    execution_time_ms=elapsed_ms,
                    syntax_valid=True,
                    error_message=f"Execution timed out after {self.timeout_sec:.1f}s in Docker sandbox",
                    test_details=[]
                )

        if proc is None:
            if self.use_docker:
                raise RuntimeError(
                    "Docker sandbox isolation was requested (use_docker=True), but Docker is unavailable "
                    "or container execution failed. Silent fallback to unsandboxed execution is prohibited (plan §33)."
                )
            with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as tmp:
                tmp.write(full_script)
                tmp_path = tmp.name

            try:
                proc = subprocess.run(
                    [self.python_exe, tmp_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec
                )
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                stdout = proc.stdout
                stderr = proc.stderr
                return_code = proc.returncode
            except subprocess.TimeoutExpired:
                elapsed_ms = self.timeout_sec * 1000.0
                return EvalResult(
                    success=False,
                    pass_rate=0.0,
                    total_tests=len(test_cases) if test_cases else 0,
                    passed_tests=0,
                    execution_time_ms=elapsed_ms,
                    syntax_valid=True,
                    error_message=f"Execution timed out after {self.timeout_sec:.1f}s",
                    test_details=[]
                )
            finally:
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

        # 4. Parse results
        if not test_cases:
            success = (return_code == 0)
            return EvalResult(
                success=success,
                pass_rate=1.0 if success else 0.0,
                total_tests=0,
                passed_tests=0,
                execution_time_ms=elapsed_ms,
                stdout=stdout,
                stderr=stderr,
                error_message="" if success else (stderr.strip() if stderr.strip() else f"Process exited with code {return_code}"),
                test_details=[]
            )

        # Parse test harness JSON
        details = []
        passed = 0
        if "__TEST_RESULTS_JSON__" in stdout:
            parts = stdout.split("__TEST_RESULTS_JSON__")
            stdout_clean = parts[0]
            try:
                details = json.loads(parts[1].strip())
                for d in details:
                    if d.get("passed", False):
                        passed += 1
            except Exception:
                pass
        else:
            stdout_clean = stdout

        total = len(test_cases)
        pass_rate = (passed / total) if total > 0 else 0.0

        return EvalResult(
            success=(passed == total and return_code == 0),
            pass_rate=pass_rate,
            total_tests=total,
            passed_tests=passed,
            execution_time_ms=elapsed_ms,
            stdout=stdout_clean,
            stderr=stderr,
            syntax_valid=True,
            error_message="" if return_code == 0 else f"Process exited with code {return_code}",
            test_details=details
        )


SandboxExecutor = CodeSandbox
