"""Generate MAS_SHT_Colab.ipynb from MAS_SHT_Kaggle.ipynb.

Only the platform-specific pieces are rewritten (title, Cell 1 setup, Cell 6
download hint, notebook metadata). Cells 2..5.5 are copied byte-for-byte so the
two notebooks stay in sync and run the identical experiment.
"""
import json

SRC = "MAS_SHT_Kaggle.ipynb"
DST = "MAS_SHT_Colab.ipynb"

TITLE_MD = """# MAS-SHT — Google Colab Experiment Harness

**Before running:**
1. **Runtime → Change runtime type → Hardware accelerator: T4 GPU**
2. Add **Secrets** (🔑 icon, left sidebar) and toggle *Notebook access* on for each:
   - `GITHUB_TOKEN` — GitHub PAT with "Contents: read" (skip if the repo is public)
   - `HF_API_KEY` — HuggingFace token (needed to download the Qwen model)
   - `GROQ_API_KEY` — only for a Groq run (not needed for the local-model run)
3. Run cells top to bottom. Re-running Cell 3 resumes from the last checkpoint.

> **Tip:** keep `USE_DRIVE = True` in Cell 1 so checkpoints land in Google Drive.
> Colab disconnects after long idle / ~12 h; if it drops, just re-run Cell 1 then
> Cell 3 and the run continues from where it stopped (no compute lost)."""

CELL1 = """# === Cell 1 — Colab Setup =====================================================
# Installs deps, clones the GitHub repo via PAT, loads Colab Secrets, and
# (optionally) mounts Google Drive so checkpoints survive a Colab disconnect.
# Re-running is safe: git pull updates the repo, pip skips installed pkgs.
#
# BEFORE RUNNING:
#   1. Runtime -> Change runtime type -> Hardware accelerator: T4 GPU
#   2. Add Secrets (key icon, left sidebar), then enable "Notebook access":
#        GITHUB_TOKEN  — GitHub PAT, "Contents: read"  (skip if repo is public)
#        HF_API_KEY    — HuggingFace token (needed to download the Qwen model)
#        GROQ_API_KEY  — Groq API key (only for a Groq run; not needed locally)

import os, sys, subprocess

# ── Persist results to Google Drive? (recommended — Colab disconnects mid-run) ──
USE_DRIVE = True   # False -> keep everything in ephemeral /content (lost on disconnect)

# ── Dependencies ──────────────────────────────────────────────────────────────
# Colab already ships torch / sympy / scipy / statsmodels / pandas / matplotlib /
# tqdm. We DO NOT reinstall torch (that breaks the CUDA build). Install the rest.
DEPS = [
    'openai>=1.40.0',
    'bitsandbytes',           # 4-bit (NF4) quant for 7B models
    'transformers>=4.44.0',   # recent enough for Qwen2.5
    'accelerate', 'huggingface_hub', 'datasets',
    'statsmodels', 'python-dotenv', 'tabulate', 'tqdm',
]
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q'] + DEPS)
print("Deps installed.")

# ── Colab Secrets ─────────────────────────────────────────────────────────────
def _get_secret(name):
    try:
        from google.colab import userdata
        try:
            return userdata.get(name)
        except Exception:
            return os.environ.get(name)
    except Exception:
        return os.environ.get(name)

_gh_token   = _get_secret('GITHUB_TOKEN')   # None -> assumes repo is public
_hf_token   = _get_secret('HF_API_KEY')
_groq_token = _get_secret('GROQ_API_KEY')

if _hf_token:
    os.environ['HF_API_KEY']        = _hf_token
    os.environ['HUGGINGFACE_TOKEN'] = _hf_token
    os.environ['HF_TOKEN']          = _hf_token
    print("HF token loaded.")
else:
    print("WARNING: HF_API_KEY not set — model download may fail for gated repos.")

if _groq_token:
    os.environ['GROQ_API_KEY'] = _groq_token
    print("Groq token loaded.")
else:
    print("Note: GROQ_API_KEY not set — fine for the local-model run; needed only for a Groq run.")

# ── Optional: mount Google Drive for checkpoint persistence ───────────────────
if USE_DRIVE:
    try:
        from google.colab import drive
        drive.mount('/content/drive')
        ROOT = '/content/drive/MyDrive/MAS_SHT'
        print("Google Drive mounted — results/checkpoints persist across disconnects.")
    except Exception as e:
        ROOT = '/content/MAS_SHT'
        print(f"Drive mount unavailable ({e}); using ephemeral {ROOT}.")
else:
    ROOT = '/content/MAS_SHT'

# ── Clone / update repo ───────────────────────────────────────────────────────
REPO_DIR  = '/content/llm_thesis'
REPO_NAME = 'marios4371/llm_thesis'

if _gh_token:
    REPO_URL = f'https://{_gh_token}@github.com/{REPO_NAME}.git'
else:
    REPO_URL = f'https://github.com/{REPO_NAME}.git'

if not os.path.isdir(REPO_DIR):
    print("Cloning repo…")
    result = subprocess.run(['git', 'clone', REPO_URL, REPO_DIR],
                            capture_output=True, text=True)
    if result.returncode != 0:
        err = result.stderr.replace(_gh_token or '', '***') if _gh_token else result.stderr
        raise RuntimeError(f"git clone failed:\\n{err}")
    print("Clone OK.")
else:
    print("Repo already present — pulling latest…")
    subprocess.run(['git', '-C', REPO_DIR, 'remote', 'set-url', 'origin', REPO_URL],
                   capture_output=True)
    result = subprocess.run(['git', '-C', REPO_DIR, 'pull', '--ff-only'],
                            capture_output=True, text=True)
    print(result.stdout.strip() or "Already up to date.")

if REPO_DIR not in sys.path:
    sys.path.insert(0, REPO_DIR)

# ── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR    = f'{ROOT}/results'
CHECKPOINT_DIR = f'{ROOT}/checkpoints'
ARTIFACTS_DIR  = f'{ROOT}/artifacts'
for d in [RESULTS_DIR, CHECKPOINT_DIR, ARTIFACTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ── GPU check ─────────────────────────────────────────────────────────────────
import torch
print(f"\\nGPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU! Runtime -> Change runtime type -> T4 GPU, then Runtime -> Restart session.")

print(f"\\nRepo    : {REPO_DIR}")
print(f"Results : {RESULTS_DIR}")
print(f"Checkpts: {CHECKPOINT_DIR}")"""


def as_source_lines(text: str):
    """Match Jupyter's storage format: list of lines, each keeping its newline
    except the last."""
    return text.splitlines(keepends=True)


def fix_broken_prints(s: str) -> str:
    """Repair print() calls whose '\\n' was saved as a literal newline inside a
    single-quoted string (a latent SyntaxError carried over from the source
    notebook's Cell 5.5)."""
    fixes = [
        ("print('\nSaved:', _out)", "print('\\nSaved:', _out)"),
        ("print('\n' + '=' * 72)", "print('\\n' + '=' * 72)"),
        ("print('\nSIV CSV tables written to', ARTIFACTS_DIR)",
         "print('\\nSIV CSV tables written to', ARTIFACTS_DIR)"),
    ]
    for bad, good in fixes:
        s = s.replace(bad, good)
    return s


def main():
    with open(SRC, encoding="utf-8") as f:
        nb = json.load(f)

    title_done = cell1_done = cell6_done = False
    for cell in nb["cells"]:
        src = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]

        if cell["cell_type"] == "markdown" and "Kaggle Experiment Harness" in src and not title_done:
            cell["source"] = as_source_lines(TITLE_MD)
            title_done = True
            continue

        if cell["cell_type"] == "code" and "Cell 1 — Kaggle Setup" in src and not cell1_done:
            cell["source"] = as_source_lines(CELL1)
            cell["outputs"] = []
            cell["execution_count"] = None
            cell1_done = True
            continue

        if cell["cell_type"] == "code" and "To download all results" in src and not cell6_done:
            new_src = src.replace(
                "print(f'\\nTo download all results: File Explorer (left panel) → /kaggle/working/MAS_SHT/')",
                "print(f'\\nAll results saved under: {ROOT}')\n"
                "print('Download them from the Files panel (folder icon, left sidebar), "
                "or open them in Google Drive if USE_DRIVE=True.')",
            )
            cell["source"] = as_source_lines(fix_broken_prints(new_src))
            cell6_done = True
            continue

        # Copy every other cell verbatim, but repair the latent Cell 5.5
        # SyntaxError if present.
        if cell["cell_type"] == "code":
            cell["source"] = as_source_lines(fix_broken_prints(src))

    # ── Notebook metadata: drop Kaggle-specific block, add Colab/GPU hints ──
    md = nb.setdefault("metadata", {})
    md.pop("kaggle", None)
    md["accelerator"] = "GPU"
    md["colab"] = {"provenance": [], "gpuType": "T4"}
    md["kernelspec"] = {"display_name": "Python 3", "name": "python3"}

    with open(DST, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"title_done={title_done} cell1_done={cell1_done} cell6_done={cell6_done}")
    print(f"Wrote {DST} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
