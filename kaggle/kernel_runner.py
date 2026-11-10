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
  smoke      a few minutes of every code path (environment check before the long runs).
"""
import json
import os
import shutil
import subprocess
import sys
import time

REPO_URL = "__REPO_URL__"
COMMIT = "__COMMIT__"
STAGE = "__STAGE__"

WORK = "/kaggle/working"
SRC = "/tmp/ezero" if os.path.isdir("/kaggle") else os.path.abspath("ezero_src")
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
SMOKE = [
    f"{PY} scripts/debug/nan_probe.py --steps 400 --mixer-eps 0 --out results/training_logs/nan_probe_eps0.json",
    f"{PY} scripts/debug/nan_probe.py --steps 400 --mixer-eps 1e-4 --out results/training_logs/nan_probe_eps1e-4.json",
    f"{PY} benchmarks/pretrain.py --variant base --steps 1500 --eval-every 250",
    f"{PY} benchmarks/run_benchmarks.py --model E-ZERO,SeasonalNaive --protocol sub16 --no-merge",
    f"{PY} benchmarks/run_efficiency.py --model E-ZERO --no-cpu",
]
# pip install steps, model ids (one virtualenv per package; --system-site-packages reuses Kaggle's torch).
# uni2ts pins torch<2.5 / numpy 1.26 / scipy 1.11, which have no Python 3.13 wheels; its code runs with
# current versions, so it is installed without dependencies plus the modules it actually imports.
TSFM = [
    (["'chronos-forecasting>=1.5'"], ["Chronos-Bolt-T", "Chronos-T"]),
    (["granite-tsfm"], ["TTM"]),
    (["--no-deps 'uni2ts>=2'", "einops jaxtyping jax gluonts lightning hydra-core python-dotenv safetensors"],
     ["Moirai-S"]),
    (["'timesfm[torch]>=2'"], ["TimesFM-2.5"]),
]
TSFM_MODELS = "__TSFM_MODELS__"  # "ALL" or a comma list rendered by build_kernels.py


def sh(cmd, cwd=None, env=None, log_name="run.log", check=False, timeout=None):
    os.makedirs(LOG, exist_ok=True)
    print(f"\n$ {cmd}", flush=True)
    t0 = time.time()
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
    for rel in ("results/cache", "results/training_logs", "data/models"):
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


def run_tsfm(env):
    status = {}
    wanted = None if TSFM_MODELS in ("ALL", "") else set(TSFM_MODELS.split(","))
    for steps, all_models in TSFM:
        models = [m for m in all_models if wanted is None or m in wanted]
        if not models:
            continue
        venv = f"/tmp/venv_{all_models[0].split('-')[0].lower()}"
        vpy = os.path.join(venv, "bin", "python")
        name = all_models[0]
        spec = " + ".join(steps)
        log_name = f"tsfm_{name}.log"
        install = make_venv(venv, log_name)
        rc = 1 if install is None else 0
        for step in steps:
            rc = rc or install(step)
        rc = rc or install(f"--no-deps -e {SRC}")
        if rc:
            status[spec] = f"install failed ({rc})"
            continue
        ids = ",".join(models)
        rc = sh(f"{vpy} benchmarks/run_benchmarks.py --model {ids} --protocol sub16 --no-merge --skip-unavailable",
                cwd=SRC, env=env, log_name=f"tsfm_{name}.log")
        rc2 = sh(f"{vpy} benchmarks/run_efficiency.py --model {ids} --skip-unavailable --cpu-reps 1 --reps 5",
                 cwd=SRC, env=env, log_name=f"tsfm_{name}.log")
        sh(f"{PY} -m pip --python {vpy} freeze > {LOG}/pip_freeze_{name}.txt", log_name=log_name)
        status[spec] = {"eval_rc": rc, "efficiency_rc": rc2}
        collect()
    return status


def main():
    t0 = time.time()
    env = setup()
    sh("nvidia-smi || true")
    sh(f"{PY} -c \"import torch;print(torch.__version__, torch.cuda.is_available())\"")
    status = {}
    if STAGE == "tsfm":
        status = run_tsfm(env)
    else:
        cmds = {"core": CORE, "ablations": ABLATIONS, "fullshot": FULLSHOT, "smoke": SMOKE}[STAGE]
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
