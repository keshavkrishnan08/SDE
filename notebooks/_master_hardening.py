"""Hardening code blocks shared by the Kaggle (08a/08b) and Colab (08) master
notebooks. Also exposes `safe_stage(name, code)` for wrapping analytical
stage cells in a try/except so a failure in one stage doesn't poison the rest.
"""


def safe_stage(name: str, code: str) -> str:
    """Wrap a stage's code in a try/except that logs the failure and lets
    subsequent cells continue.

    Critical infrastructure (setup, SHARED_CODE, LOAD_DATA, STAGE 0, POST
    verify, ZIP) should NOT be wrapped — losing those means the whole run
    is degenerate. Analytic / figure / table stages SHOULD be wrapped so a
    single bug doesn't take down the entire 08b run."""
    # Indent every line of `code` by 4 spaces so it sits inside `try:`.
    indented = "\n".join(("    " + ln if ln else "") for ln in code.splitlines())
    return (
        f"# ==== SAFE STAGE: {name} ====\n"
        f"import traceback as _tb_safe_stage\n"
        f"try:\n"
        f"{indented}\n"
        f"except Exception as _e_safe_stage:\n"
        f"    print('\\n' + '!' * 70)\n"
        f"    print(f'[STAGE FAILED] {name}: "
        f"{{type(_e_safe_stage).__name__}}: {{_e_safe_stage}}')\n"
        f"    _tb_safe_stage.print_exc()\n"
        f"    print(f'[STAGE FAILED] {name} skipped — continuing.')\n"
        f"    print('!' * 70 + '\\n')\n"
    )


# ============================================================
# PREFLIGHT_SANITY_CODE — runs right after FAST_START
#   - Asserts every name that downstream cells consume is in scope
#   - Validates any pre-existing checkpoint is loadable + not NaN
#   - Reports free disk so the user catches an unwriteable session early
# ============================================================
PREFLIGHT_SANITY_CODE = '''\
# ==== PREFLIGHT — sanity check before any heavy work ====
import shutil as _shutil
_required_names = [
    # Gates + checkpoint paths from FAST_START
    "NEED_NB2_TRAINING", "SDE_CKPT", "SCORE_CKPT",
    "HAVE_VAE", "HAVE_SPLITS", "HAVE_LATENTS",
    "HAVE_KT", "HAVE_PHYS", "HAVE_EXTENDED", "HAVE_EXT",
    "NEED_GOLDEN_RETRAIN", "NEED_IMAGE_FEATURES",
    # Paths + device from setup
    "DEVICE", "DATA_DIR", "WORK_DIR", "PERSIST_DIR",
    "CHECKPOINT_DIR", "LATENT_DIR", "SPLITS_DIR",
    "EXTENDED_DIR", "RESULTS_DIR", "FIGURES_DIR",
    # Imports that STAGE0 / training / BASELINES blocks need in scope
    "Dataset", "DataLoader", "torch", "nn", "F", "np", "pd", "tqdm",
    "time", "gc", "math",
]
_missing = [n for n in _required_names if n not in globals()]
if _missing:
    raise NameError(
        f"PREFLIGHT FAIL: setup/fast-start cells did not publish required "
        f"names: {_missing}. Re-run the SETUP and FAST_START cells, or pull "
        f"the latest notebook from GitHub.")

# Validate any existing checkpoints are loadable and not NaN-poisoned.
_corrupt = []
for _name, _p in [("vae",   CHECKPOINT_DIR / "vae_best.pt"),
                  ("sde",   SDE_CKPT),
                  ("score", SCORE_CKPT)]:
    if not _p.exists():
        continue
    try:
        _sd = torch.load(_p, map_location="cpu", weights_only=False)
        _bad = [k for k, v in _sd.items()
                if torch.is_tensor(v) and not torch.isfinite(v).all()]
        if _bad:
            _corrupt.append((_name, _p, f"NaN/Inf in tensors: {_bad[:3]}"))
    except Exception as _e:
        _corrupt.append((_name, _p, f"torch.load failed: {_e}"))
if _corrupt:
    for _n, _p, _r in _corrupt:
        print(f"  [WARN] corrupt checkpoint {_n} at {_p}: {_r} — deleting.")
        try: _p.unlink()
        except Exception: pass
    NEED_NB2_TRAINING = not (SDE_CKPT.exists() and SCORE_CKPT.exists())
    print(f"  [INFO] NEED_NB2_TRAINING re-evaluated to {NEED_NB2_TRAINING}.")

# Disk-space sanity (Kaggle gives ~20 GB; warn under 5 GB free)
try:
    _free_gb = _shutil.disk_usage(str(WORK_DIR)).free / 1e9
    print(f"  Free disk @ WORK_DIR : {_free_gb:.1f} GB")
    if _free_gb < 5:
        print(f"  [WARN] Less than 5 GB free — training/checkpoint saves may fail.")
except Exception:
    pass
print("PREFLIGHT: all required names present.")
'''


# ============================================================
# POST_STAGE0_VERIFY_CODE — runs immediately after STAGE0_CODE
#   - Confirms sde_best.pt + score_best.pt exist (or that STAGE0 took
#     its skip branch with pre-existing ckpts)
#   - Loads each ckpt and asserts no NaN/Inf weights crept in during
#     150-epoch training. Bad ckpts are deleted so the next run retries.
# ============================================================
POST_STAGE0_VERIFY_CODE = '''\
# ==== Verify STAGE 0 produced healthy checkpoints ====
_missing = [p.name for p in (SDE_CKPT, SCORE_CKPT) if not p.exists()]
if _missing:
    raise RuntimeError(
        f"STAGE 0 finished but checkpoints missing: {_missing}. "
        f"This usually means training crashed or was interrupted. "
        f"Re-run STAGE 0 (it auto-resumes).")

_bad = []
for _name, _p in [("sde", SDE_CKPT), ("score", SCORE_CKPT)]:
    _sd = torch.load(_p, map_location="cpu", weights_only=False)
    for k, v in _sd.items():
        if torch.is_tensor(v) and not torch.isfinite(v).all():
            _bad.append((_name, k)); break
if _bad:
    for _n, _k in _bad:
        _p = SDE_CKPT if _n == "sde" else SCORE_CKPT
        print(f"  [FAIL] NaN/Inf in {_n} ckpt tensor {_k!r} — deleting {_p}")
        _p.unlink()
    raise RuntimeError(
        f"STAGE 0 trained corrupt checkpoints (NaN weights). "
        f"Bad ckpts deleted. Re-run STAGE 0 — usual fix is to lower lr "
        f"(currently 1e-4 for SDE, 2e-4 for Score) or check for NaN in "
        f"data/Z_MEAN/Z_STD.")
print("[OK] STAGE 0 checkpoints verified (no NaN/Inf weights).")
'''


# ============================================================
# STAGE_M1_SAFE_FALLBACK_CODE — runs before STAGE_MINUS1_CODE
#   - If image_features.npy already exists, no-op (STAGE_MINUS1 will skip)
#   - Else if Golden retrain extracted images into DATA_DIR/cloudcv,
#     symlink them into WORK_DIR/cloudcv where STAGE_MINUS1 expects them
#   - Else (no images anywhere) write zero-filled image_features.npy so
#     STAGE_MINUS1 takes its skip branch instead of attempting downloads
#     with the broken data.nlr.gov URLs hardcoded inside STAGE_MINUS1_CODE
# ============================================================
STAGE_M1_SAFE_FALLBACK_CODE = '''\
# ==== Pre-STAGE -1 fallback (path-fix + safe skip) ====

# --- Zero: if NEED_IMAGE_FEATURES is True (we want real features) and any
# existing zero-fill image_features.npy is on disk, delete it so STAGE -1
# can extract real features from the just-downloaded CloudCV images.
if globals().get("NEED_IMAGE_FEATURES", False):
    for _s in ["train", "val", "test"]:
        _f = LATENT_DIR / f"{_s}_image_features.npy"
        if _f.exists():
            _arr = np.load(_f, mmap_mode="r")
            if _arr.size > 0 and not _arr.any():
                print(f"  [REFRESH] Deleting zero-fill {_f.name} so STAGE -1 can extract real features.")
                _f.unlink()

# --- First: detect stale zero-fill image_features.npy that would inflate
# cov dim past a previously-saved SDE checkpoint's training-time c_dim.
# If we wrote zero-fill features on a prior run BEFORE STAGE 0 trained, the
# ckpt expects (cov+phys+img) dim. If we wrote them AFTER STAGE 0 trained
# without images, the ckpt expects (cov+phys) only, and loading it against
# the larger cov will crash with shape-mismatch. Detect + delete those.
if SDE_CKPT.exists():
    try:
        _sd = torch.load(SDE_CKPT, map_location="cpu", weights_only=False)
        _first_w_key = next(k for k in _sd if k.endswith("drift.net.0.weight"))
        _ckpt_input_dim = _sd[_first_w_key].shape[1]
        _ckpt_c_dim = _ckpt_input_dim - 64 - 1   # z_dim=64, time=1
        for _s in ["train", "val", "test"]:
            _f = LATENT_DIR / f"{_s}_image_features.npy"
            if not _f.exists(): continue
            _arr = np.load(_f)
            if _arr.size == 0 or not (_arr == 0).all():
                continue   # real (non-zero) features — leave them alone
            _orig = np.load(LATENT_DIR / f"{_s}_covariates.npy")
            _phys = np.load(LATENT_DIR / f"{_s}_physics_features.npy")
            _base_dim = _orig.shape[1] + _phys.shape[1]
            _with_img = _base_dim + _arr.shape[1]
            if _ckpt_c_dim == _base_dim and _ckpt_c_dim != _with_img:
                print(f"  [FIX] Deleting stale zero-fill {_f.name} "
                      f"(ckpt c_dim={_ckpt_c_dim}, would inflate to {_with_img}).")
                _f.unlink()
            elif _ckpt_c_dim != _base_dim and _ckpt_c_dim != _with_img:
                print(f"  [WARN] {_f.name}: ckpt c_dim={_ckpt_c_dim} matches "
                      f"neither {_base_dim} nor {_with_img}. Leaving in place.")
        del _sd
    except Exception as _e:
        print(f"  [WARN] could not infer ckpt c_dim ({_e}); skipping zero-fill check.")

_have_feats = all((LATENT_DIR / f"{s}_image_features.npy").exists()
                  for s in ["train", "val", "test"])
if _have_feats:
    print("[OK] image_features.npy present for all splits — STAGE -1 will skip.")
else:
    _data_cv  = DATA_DIR / "cloudcv"   # Golden-retrain writes here
    _work_cv  = WORK_DIR / "cloudcv"   # STAGE_MINUS1_CODE reads here
    _data_imgs = list(_data_cv.glob("2019_*/images/*.jpg")) if _data_cv.exists() else []
    _work_imgs = list(_work_cv.glob("2019_*/images/*.jpg")) if _work_cv.exists() else []
    if _work_imgs:
        print(f"[OK] STAGE -1 will find {len(_work_imgs)} images at WORK_DIR/cloudcv.")
    elif _data_imgs:
        print(f"[FIX] Bridging path mismatch: {len(_data_imgs)} images in "
              f"DATA_DIR/cloudcv — symlinking into WORK_DIR/cloudcv ...")
        _work_cv.mkdir(parents=True, exist_ok=True)
        for _day in _data_cv.iterdir():
            if _day.is_dir():
                _dst = _work_cv / _day.name
                if not _dst.exists():
                    try:
                        _dst.symlink_to(_day.resolve())
                    except Exception:
                        import shutil as _sh
                        _sh.copytree(_day, _dst)
        print("[FIX] Symlinks created.")
    else:
        print("[WARN] No CloudCV images found in DATA_DIR/cloudcv or "
              "WORK_DIR/cloudcv — writing zero-fill image_features.npy so "
              "STAGE -1 skips. Paper will report 0 image features.")
        for _split in ["train", "val", "test"]:
            _n = len(pd.read_parquet(SPLITS_DIR / f"{_split}.parquet"))
            np.save(LATENT_DIR / f"{_split}_image_features.npy",
                    np.zeros((_n, 10), dtype=np.float32))
        print("[OK] zero-fill image_features.npy written for all splits.")
'''
