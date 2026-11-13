"""Kaggle kernel entry point (copied into every kernel folder by kaggle/build_kernels.py).

The kernel clones this repository at a pinned commit, installs it, runs the commands of
one STAGE and copies every produced artifact into /kaggle/working/outputs/ using the
repository's own paths (results/cache/..., results/training_logs/..., data/models/...),
so ``kaggle/fetch_outputs.py`` can drop them straight back into the repo.

Stages
  core       pre-train E-ZERO (base), evaluate E-ZERO + naive baselines (full & sub16 protocols,
             train-split and legacy full-series scaling), efficiency.
  ablations  pre-train the 7 ablation variants (thesis Table 5) and evaluate them.
  tsfm       zero-shot foundation-model baselines, each package in its own virtualenv.
  fullshot   DLinear / PatchTST / iTransformer trained per dataset and horizon.
  fullshot2  TSMixer / TimeMixer / TimesNet;  fullshot3  FEDformer / Autoformer (same protocol).
  smoke      a few minutes of every code path (environment check before the long runs).
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import time

REPO_URL = "__REPO_URL__"
COMMIT = "__COMMIT__"
STAGE = "__STAGE__"

# Kaggle writes into /kaggle/working. Elsewhere (Colab, a workstation) EZERO_WORK and
# EZERO_SRC say where to put the outputs and where the checkout is; see colab/run_stage.py.
WORK = os.environ.get("EZERO_WORK") or "/kaggle/working"
SRC = os.environ.get("EZERO_SRC") or ("/tmp/ezero" if os.path.isdir("/kaggle") else os.path.abspath("ezero_src"))
OUT = os.path.join(WORK, "outputs")
LOG = os.path.join(WORK, "logs")

PY = sys.executable
CORE = [
    f"{PY} benchmarks/pretrain.py --variant base --time-budget-min 240",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO,group:naive --protocol full --no-merge",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO,group:naive --protocol sub16 --no-merge",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO,group:naive --protocol full --scaler full --no-merge",
    f"{PY} benchmarks/run_efficiency.py --model E-ZERO,group:naive",
]
ABLATIONS = [
    f"{PY} benchmarks/pretrain.py --variant noMRC,noDTP,noRAE,noCRRB,noExog,noTCN,noNBEATS --time-budget-min 75",
    f"{PY} benchmarks/run_benchmarks.py --model group:ezero-ablation --protocol full --no-merge --skip-unavailable",
    f"{PY} benchmarks/run_benchmarks.py --model group:ezero-ablation --protocol sub16 --no-merge --skip-unavailable",
    f"{PY} benchmarks/run_efficiency.py --model group:ezero-ablation --no-cpu --skip-unavailable",
]
FULLSHOT = [
    f"{PY} benchmarks/run_benchmarks.py --model DLinear,iTransformer,PatchTST --protocol full --no-merge "
    "--skip-unavailable --fullshot-max-steps 2000 --fullshot-max-epochs 10",
    f"{PY} benchmarks/run_efficiency.py --model DLinear,iTransformer,PatchTST --skip-unavailable",
]
FULLSHOT2 = [
    f"{PY} benchmarks/run_benchmarks.py --model TSMixer,TimeMixer,TimesNet --protocol full --no-merge "
    "--skip-unavailable --fullshot-max-steps 2000 --fullshot-max-epochs 10",
    f"{PY} benchmarks/run_efficiency.py --model TSMixer,TimeMixer,TimesNet --skip-unavailable",
]
FULLSHOT3 = [
    f"{PY} benchmarks/run_benchmarks.py --model FEDformer,Autoformer --protocol full --no-merge "
    "--skip-unavailable --fullshot-max-steps 2000 --fullshot-max-epochs 10",
    f"{PY} benchmarks/run_efficiency.py --model FEDformer,Autoformer --skip-unavailable",
]
# fullval: full-shot baselines with their official per-dataset settings (baselines/paper_configs.py),
# 3 seeds on ETTh1 and 1 seed on ETTm1, plus DLinear at its paper lookback on every dataset.
_RB = f"{PY} benchmarks/run_benchmarks.py --protocol full --no-merge --skip-unavailable"
FULLVAL = []
for _ds, _seeds in (("ETTh1", (0, 1, 2)), ("ETTm1", (0,))):
    for _seed in _seeds:
        FULLVAL += [
            f"{_RB} --paper-config --seed {_seed} --datasets {_ds} --context-length 336 --model DLinear,PatchTST",
            f"{_RB} --paper-config --seed {_seed} --datasets {_ds} --context-length 512 --model TSMixer",
            f"{_RB} --paper-config --seed {_seed} --datasets {_ds} --context-length 96 "
            "--model iTransformer,TimesNet,TimeMixer,FEDformer,Autoformer",
        ]
FULLVAL.append(f"{_RB} --context-length 336 --model DLinear --fullshot-max-steps 1000000000 --fullshot-max-epochs 10")
# tweaks: E-ZERO v3 candidates (configs/ezero_v3{a,b,c}.yaml on the extended corpus), scored on the
# zero-shot development benchmark, automatic pre-registered selection (benchmarks/select_v3.py), then a
# single evaluation of the selected model (and of the v2 checkpoint with flip averaging) on the LSF data.
_EXT = "--corpus configs/pretrain_corpus_extended.yaml"
TWEAKS = [
    # N-BEATS design study at the thesis recipe (same corpus/budget/seed as the ablation variants):
    # linear shortcut + N-BEATS + TCN, comparable with E-ZERO (N-BEATS + TCN) and E-ZERO-noNBEATS (linear + TCN)
    f"{PY} benchmarks/pretrain.py --variant base --config configs/ezero_skip.yaml --name ezero_skip --time-budget-min 75",
] + [
    f"{PY} benchmarks/pretrain.py --variant base --config configs/{c}.yaml {_EXT} --name {c} --time-budget-min 75"
    for c in ("ezero_v3a", "ezero_v3b", "ezero_v3c", "ezero_v3d")
] + [
    # candidates for E-ZERO-v3: every one keeps the N-BEATS stack
    f"{PY} benchmarks/dev_eval.py --checkpoint data/models/ezero_base.pt data/models/ezero_skip_base.pt "
    "data/models/ezero_v3a_base.pt data/models/ezero_v3b_base.pt data/models/ezero_v3c_base.pt "
    "data/models/ezero_v3d_base.pt --tta none flip --label v3-candidates --baselines",
    # reference only (not eligible): the linear + TCN ablation variant
    f"{PY} benchmarks/dev_eval.py --checkpoint data/models/ezero_noNBEATS.pt --tta none flip --label v3-reference",
    f"{PY} benchmarks/select_v3.py --label v3-candidates",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO-skip,E-ZERO-v3,E-ZERO-flip --protocol full --no-merge "
    "--skip-unavailable",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO-skip,E-ZERO-v3,E-ZERO-flip --protocol sub16 --no-merge "
    "--skip-unavailable",
    f"{PY} benchmarks/run_efficiency.py --model E-ZERO,E-ZERO-skip,E-ZERO-v3,E-ZERO-flip --skip-unavailable",
]
# faithful: the components described in thesis Sec. 3.3.3-3.3.5 but missing from the 2024 code
# (feature attention, conditional convolutions, similarity matching + adaptive inference). Same
# corpus, optimiser, budget and seed as the ablation variants, so E-ZERO-faithful is directly
# comparable with E-ZERO. Each component is then ablated. The development benchmark decides
# whether the faithful model replaces the thesis configuration; the LSF data is scored once.
FAITHFUL = [
    f"{PY} benchmarks/pretrain.py --variant base,faithNoAttn,faithNoCond,faithNoSim "
    "--config configs/ezero_faithful.yaml --name ezero_faithful --time-budget-min 75",
    f"{PY} benchmarks/dev_eval.py --checkpoint data/models/ezero_faithful_base.pt "
    "data/models/ezero_faithful_faithNoAttn.pt data/models/ezero_faithful_faithNoCond.pt "
    "data/models/ezero_faithful_faithNoSim.pt data/models/ezero_base.pt "
    "--tta none flip --label faithful --baselines",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO-faithful,E-ZERO-faithNoAttn,E-ZERO-faithNoCond,"
    "E-ZERO-faithNoSim --protocol full --no-merge --skip-unavailable",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO-faithful,E-ZERO-faithNoAttn,E-ZERO-faithNoCond,"
    "E-ZERO-faithNoSim --protocol sub16 --no-merge --skip-unavailable",
    f"{PY} benchmarks/run_efficiency.py --model E-ZERO,E-ZERO-faithful --skip-unavailable",
]
# tsfmfix: the two foundation models the first tsfm run did not deliver.
#  * Moirai-S: re-run after the sample-aggregation fix (median, as its authors evaluate; the mean
#    of a heavy-tailed predictive distribution gave MSE up to 1e8 - notes section 12).
#  * Lag-Llama: the first run was cut off by Kaggle's 12 h cap after 5 of 7 datasets. It costs
#    ~54 min per ETT dataset and 2.7 h for Weather (21 channels); Electricity (321) and Traffic
#    (862) would need an estimated 40+ h each, so they are left out and reported as not attempted.
TSFM_FIX = [
    dict(models=["Moirai-S"], steps=["--no-deps 'uni2ts>=2'",
                                     "einops jaxtyping jax gluonts lightning hydra-core python-dotenv safetensors"]),
    dict(models=["Lag-Llama"], python="3.11", clone=LAG_LLAMA_REPO, opt_in=True,
         datasets="ETTh1 ETTh2 ETTm1 ETTm2 Weather",
         steps=["torch==2.4.1", "'gluonts[torch]==0.14.4' 'numpy<2' 'pandas<2.2' 'lightning<2.5' 'setuptools<75' "
                "huggingface_hub scipy pyyaml"]),
]
SMOKE = [
    f"{PY} scripts/debug/nan_probe.py --steps 400 --mixer-eps 0 --out results/training_logs/nan_probe_eps0.json",
    f"{PY} scripts/debug/nan_probe.py --steps 400 --mixer-eps 1e-4 --out results/training_logs/nan_probe_eps1e-4.json",
    f"{PY} benchmarks/pretrain.py --variant base --steps 1500 --eval-every 250",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO,SeasonalNaive --protocol sub16 --no-merge",
    f"{PY} benchmarks/run_efficiency.py --model E-ZERO --no-cpu",
]
# One virtualenv per foundation-model package.
#  * python=None  -> venv on Kaggle's interpreter with --system-site-packages (reuses Kaggle's torch/CUDA)
#  * python="3.11" -> uv-managed CPython 3.11 with its own torch 2.4.1 (CUDA 12.1 wheels): the thesis-era
#    TimesFM-1.0 (timesfm 1.x) and Lag-Llama (gluonts 0.14) have no builds for Kaggle's Python 3.13
# uni2ts pins torch<2.5 / numpy 1.26 / scipy 1.11 (no Python 3.13 wheels); its code runs with current
# versions, so it is installed without dependencies plus the modules it actually imports.
LAG_LLAMA_REPO = ("https://github.com/time-series-foundation-models/lag-llama.git",
                  "df7531a83a19b3c6a0222d703ca9bf59ef7a6ab9")
TSFM = [
    dict(models=["Chronos-T", "Chronos-Bolt-T"], steps=["'chronos-forecasting>=1.5'"]),
    dict(models=["TTM-R1", "TTM-R2"], steps=["granite-tsfm"]),
    dict(models=["Moirai-S"], steps=["--no-deps 'uni2ts>=2'",
                                     "einops jaxtyping jax gluonts lightning hydra-core python-dotenv safetensors"]),
    dict(models=["TimesFM-2.5"], steps=["'timesfm[torch]>=2'"]),
    # timesfm 1.2.6[torch] also pulls jax[cuda12] and wandb (several GB); the PyTorch path only needs a CPU jax
    # for an import in its covariates module. Tested locally: 203.6M parameters, Python 3.11, torch 2.4.1.
    dict(models=["TimesFM-1.0"], python="3.11",
         steps=["torch==2.4.1", "--no-deps timesfm==1.2.6",
                "absl-py einshape huggingface_hub utilsforecast 'numpy<2' pandas scikit-learn pyyaml 'jax<0.5' 'jaxlib<0.5'"]),
    # Lag-Llama is not a pip package: it runs the authors' GitHub code (pinned commit). Opt-in only - it runs
    # when "Lag-Llama" is named explicitly in TSFM_MODELS, never with "ALL".
    # thesis Table 3/4 only (no zero-shot forecasting head): timed, never scored for accuracy
    dict(models=["GPT4TS"], steps=[], efficiency_only=True),
    dict(models=["MOMENT"], python="3.11", steps=["torch==2.4.1", "momentfm pandas pyyaml"], efficiency_only=True),
    dict(models=["Lag-Llama"], python="3.11", clone=LAG_LLAMA_REPO, opt_in=True,
         steps=["torch==2.4.1", "'gluonts[torch]==0.14.4' 'numpy<2' 'pandas<2.2' 'lightning<2.5' 'setuptools<75' "
                "huggingface_hub scipy pyyaml"]),
]
TSFM_MODELS = "__TSFM_MODELS__"  # "ALL" or a comma list rendered by build_kernels.py

# tsfmval stage: the protocols the models' own / related papers used (reproduction check).
# full = every sliding test window (TTM Tables 11/13, Moirai Table 22); last = last test window
# (TTM Table 15, TimesFM Table 5). Moirai excludes Traffic (in its pre-training data).
_ALL = "ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity Traffic"
_NO_TRAFFIC = "ETTh1 ETTh2 ETTm1 ETTm2 Weather Electricity"
TSFM_VAL = {
    "TTM-R1": [("full", _ALL), ("last", _ALL)],
    "TTM-R2": [("full", _ALL)],
    "Moirai-S": [("full", _NO_TRAFFIC)],
    "TimesFM-1.0": [("full", _NO_TRAFFIC), ("last", "ETTh1 ETTh2 ETTm1 ETTm2")],
    "Chronos-T": [("last", _ALL)],
    "Lag-Llama": [("last", _ALL)],
}


# Kaggle stops a session after 12 h and may discard its outputs. Every command therefore runs under a
# shared deadline (10.5 h after start): when it is reached the command is killed, later commands are
# skipped (exit code 124) and whatever was produced so far is still collected and saved.
DEADLINE = time.time() + float(os.environ.get("EZERO_DEADLINE_H") or 10.5) * 3600


def sh(cmd, cwd=None, env=None, log_name="run.log", check=False, timeout=None):
    os.makedirs(LOG, exist_ok=True)
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
    remaining = int(DEADLINE - t0)
    if remaining < 60:
        print("[skipped: stage deadline reached]", flush=True)
        with open(os.path.join(LOG, log_name), "a") as log:
            log.write(f"\n$ {cmd}\n[skipped: stage deadline reached]\n")
        if check:
            raise SystemExit(124)
        return 124
    cmd = f"timeout -k 30 {remaining} bash -c {shlex.quote(cmd)}"
    with open(os.path.join(LOG, log_name), "a") as log:
        log.write(f"\n$ {cmd}\n")
        log.flush()
        proc = subprocess.Popen(cmd, shell=True, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        rc = proc.wait(timeout=timeout)
        log.write(f"[exit {rc} after {time.time() - t0:.0f}s]\n")
    print(f"[exit {rc} after {time.time() - t0:.0f}s]", flush=True)
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


def setup():
    os.makedirs(os.path.dirname(SRC), exist_ok=True)
    if not os.path.isdir(SRC):
        sh(f"git clone --quiet {REPO_URL} {SRC}", check=True)
    sh(f"git -C {SRC} fetch --quiet origin {COMMIT} || true")
    sh(f"git -C {SRC} checkout --quiet {COMMIT}", check=True)
    sh(f"{PY} -m pip install --quiet --no-input -e {SRC}", check=True)
    env = os.environ.copy()
    env["EZERO_GIT_COMMIT"] = COMMIT
    env["EZERO_LOTSA_DIR"] = "/tmp/lotsa"
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    return env


def collect():
    """Copy artifacts to /kaggle/working/outputs/<repo path>."""
    for rel in ("results/cache", "results/training_logs", "results/dev", "data/models"):
        src = os.path.join(SRC, rel)
        if not os.path.isdir(src):
            continue
        for root, _, files in os.walk(src):
            for f in files:
                if f.startswith("."):
                    continue
                p = os.path.join(root, f)
                dst = os.path.join(OUT, os.path.relpath(p, SRC))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(p, dst)
    for f in ("results/failures.json",):
        p = os.path.join(SRC, f)
        if os.path.exists(p):
            dst = os.path.join(OUT, f)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(p, dst)


def make_venv(venv, log_name):
    """Virtualenv that sees the system site-packages (Kaggle's torch/CUDA stack).

    Kaggle's /usr/bin/python3 has no ensurepip, so the venv is created without pip and
    packages are installed into it by the system pip (``pip --python``), or by uv as a fallback.
    Returns a function ``install(args) -> exit code``.
    """
    vpy = os.path.join(venv, "bin", "python")
    rc = sh(f"{PY} -m venv --without-pip --system-site-packages {venv}", log_name=log_name)
    if rc == 0 and sh(f"{PY} -m pip --python {vpy} --version", log_name=log_name) == 0:
        return lambda args: sh(f"{PY} -m pip --python {vpy} install --quiet --no-input {args}", log_name=log_name)
    sh(f"{PY} -m pip install --quiet uv", log_name=log_name)
    shutil.rmtree(venv, ignore_errors=True)
    if sh(f"{PY} -m uv venv --system-site-packages --python {PY} {venv}", log_name=log_name):
        return None
    return lambda args: sh(f"{PY} -m uv pip install --quiet --python {vpy} {args}", log_name=log_name)


def make_venv311(venv, log_name):
    """uv-managed CPython 3.11 virtualenv (no system site-packages). Returns install(args) -> rc."""
    vpy = os.path.join(venv, "bin", "python")
    rc = sh(f"{PY} -m pip install --quiet uv", log_name=log_name)
    rc = rc or sh(f"{PY} -m uv python install 3.11", log_name=log_name)
    rc = rc or sh(f"{PY} -m uv venv --python 3.11 {venv}", log_name=log_name)
    if rc:
        return None
    return lambda args: sh(f"{PY} -m uv pip install --quiet --python {vpy} {args}", log_name=log_name)


def run_tsfm(env):
    status = {}
    wanted = None if TSFM_MODELS in ("ALL", "") else set(TSFM_MODELS.split(","))
    for entry in (TSFM_FIX if STAGE == "tsfmfix" else TSFM):
        all_models, steps = entry["models"], entry["steps"]
        if entry.get("opt_in") and (wanted is None or not (set(all_models) & wanted)):
            continue
        models = [m for m in all_models if wanted is None or m in wanted]
        if not models:
            continue
        name = all_models[0]
        venv = f"/tmp/venv_{name.lower().replace('.', '')}"
        vpy = os.path.join(venv, "bin", "python")
        spec = " + ".join(steps)
        log_name = f"tsfm_{name}.log"
        install = make_venv311(venv, log_name) if entry.get("python") else make_venv(venv, log_name)
        rc = 1 if install is None else 0
        for step in steps:
            rc = rc or install(step)
        rc = rc or install(f"--no-deps -e {SRC}")
        run_env = dict(env)
        if entry.get("clone"):
            url, rev = entry["clone"]
            dst = f"/tmp/{name.lower()}_src"
            rc = rc or sh(f"git clone --quiet {url} {dst} && git -C {dst} checkout --quiet {rev}", log_name=log_name)
            run_env["PYTHONPATH"] = dst + os.pathsep + env.get("PYTHONPATH", "")
        if rc:
            status[spec] = f"install failed ({rc})"
            continue
        ids = ",".join(models)
        rc, rc2 = 0, 0
        if STAGE == "tsfmval":
            for m in models:
                for prot, dsets in TSFM_VAL.get(m, []):
                    rc = rc or sh(f"{vpy} benchmarks/run_benchmarks.py --model {m} --protocol {prot} --datasets {dsets} "
                                  "--no-merge --skip-unavailable", cwd=SRC, env=run_env, log_name=log_name)
                    collect()
        else:
            if not entry.get("efficiency_only"):
                dsets = f" --datasets {entry['datasets']}" if entry.get("datasets") else ""
                rc = sh(f"{vpy} benchmarks/run_benchmarks.py --model {ids} --protocol sub16{dsets} "
                        "--no-merge --skip-unavailable", cwd=SRC, env=run_env, log_name=log_name)
            rc2 = sh(f"{vpy} benchmarks/run_efficiency.py --model {ids} --skip-unavailable --cpu-reps 1 --reps 5",
                     cwd=SRC, env=run_env, log_name=log_name)
        sh(f"{PY} -m uv pip freeze --python {vpy} > {LOG}/pip_freeze_{name}.txt 2>/dev/null || "
           f"{PY} -m pip --python {vpy} freeze > {LOG}/pip_freeze_{name}.txt", log_name=log_name)
        status[spec] = {"eval_rc": rc, "efficiency_rc": rc2}
        collect()
    return status


def main():
    t0 = time.time()
    env = setup()
    sh("nvidia-smi || true")
    sh(f"{PY} -c \"import torch;print(torch.__version__, torch.cuda.is_available())\"")
    status = {}
    if STAGE in ("tsfm", "tsfmval", "tsfmfix"):
        status = run_tsfm(env)
    else:
        cmds = {"core": CORE, "ablations": ABLATIONS, "fullshot": FULLSHOT, "fullshot2": FULLSHOT2,
                "fullshot3": FULLSHOT3, "fullval": FULLVAL, "tweaks": TWEAKS,
                "faithful": FAITHFUL, "smoke": SMOKE}[STAGE]
        for c in cmds:
            status[c] = sh(c, cwd=SRC, env=env, log_name=f"{STAGE}.log")
            collect()
    collect()
    with open(os.path.join(WORK, "stage_status.json"), "w") as f:
        json.dump({"stage": STAGE, "commit": COMMIT, "status": status,
                   "minutes": round((time.time() - t0) / 60, 1)}, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
