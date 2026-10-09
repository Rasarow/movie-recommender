import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parent
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Run setup.py with 64-bit Python 3.12.")
python = root / ".venv/Scripts/python.exe"
if not python.exists():
    subprocess.run([sys.executable, "-m", "venv", str(root / ".venv")], check=True)
subprocess.run(
    [
        str(python),
        "-c",
        "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8, 'Use 64-bit Python 3.12.'",
    ],
    check=True,
)
subprocess.run(
    [str(python), "-m", "pip", "install", "-r", str(root / "requirements.txt")],
    cwd=root,
    check=True,
)
for name in ("prepare.py", "fit.py", "evaluate.py", "demo.py"):
    subprocess.run([str(python), str(root / name)], cwd=root, check=True)
print("Ready. Run run.cmd demo.py.")
