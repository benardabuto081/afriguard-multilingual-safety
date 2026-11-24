"""Write a concise environment report to docs/environment.md (CPU-only; no GPU checks)."""
import importlib.metadata as md
import platform
import shutil
import sys
from pathlib import Path


def mem_gb():
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal"):
                return round(int(line.split()[1]) / 1024 / 1024, 1)
    except OSError:
        return None

def cpu_model():
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        return platform.processor()

du = shutil.disk_usage(".")
pkgs = ["numpy", "pandas", "pyyaml", "scikit-learn", "pytest", "ruff", "torch", "transformers", "peft"]
def ver(p):
    try:
        return md.version(p)
    except md.PackageNotFoundError:
        return "not installed"

lines = [
    "# Environment report", "",
    f"- OS: {platform.platform()}",
    f"- CPU: {cpu_model()}",
    f"- Python: {sys.version.split()[0]}",
    f"- System RAM (GB): {mem_gb()}",
    f"- Disk free / total (GB): {du.free // 2**30} / {du.total // 2**30}",
    "- GPU: none (CPU-only laptop; training to run on external GPU host)", "",
    "## Packages",
] + [f"- {p}: {ver(p)}" for p in pkgs]
out = Path("docs/environment.md")
out.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
