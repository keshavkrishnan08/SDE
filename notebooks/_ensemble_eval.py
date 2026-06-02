"""Dual-architecture training glue + ensemble + champion selection (notebook 11).

The final publication notebook trains BOTH architectures on the same data:
  A. Closed-form  (_solarsde_v2.TemporalLatentSDE)   - Mixture-of-OU marginals
  B. Rollout      (_solarsde_rollout.RolloutLatentSDE) - Euler-Maruyama latent rollout
and builds a third variant:
  C. Ensemble     - pools MC samples from A and B (equal weight)

Champion selection happens ON VALIDATION (never test): the variant with the
lowest val CRPS at the SkyGPT horizon (15 min) becomes the headline model, and
the downstream suite (ablations, stratified, CV, economics, ...) runs on it.
All three variants are reported side-by-side on the SkyGPT exact benchmark so
nothing is hidden.

CODE blocks:
  STASH_CLOSEDFORM_CODE   - after closed-form STAGE 0: stash ckpt/results under closedform_* names
  STASH_ROLLOUT_CODE      - after rollout STAGE 0: stash ckpt/results under rollout_* names
  CHAMPION_SELECT_CODE    - evaluate A/B/ensemble on VAL, pick champion, restore its
                            artifacts as the canonical mdn_v2_best.pt / main results
  SKYGPT_TRIPLE_BENCHMARK_CODE - SkyGPT exact test for all 3 variants + published numbers
"""


STASH_CLOSEDFORM_CODE = '''\
# ==== Stash the closed-form model's artifacts under closedform_* names ====
import shutil as _sh
_CF_CKPT = CHECKPOINT_DIR / "mdn_closedform_best.pt"
_CF_RES  = RESULTS_DIR / "closedform_main_results.csv"
if (CHECKPOINT_DIR / "mdn_v2_best.pt").exists():
    _sh.copy(CHECKPOINT_DIR / "mdn_v2_best.pt", _CF_CKPT)
    print(f"  stashed closed-form ckpt -> {_CF_CKPT.name}")
if (RESULTS_DIR / "solar_sde_main_results.csv").exists():
    _sh.copy(RESULTS_DIR / "solar_sde_main_results.csv", _CF_RES)
    print(f"  stashed closed-form results -> {_CF_RES.name}")
# Remove canonical names so the rollout STAGE 0 trains fresh instead of skipping.
for _f in ["mdn_v2_best.pt", "sde_best.pt", "score_best.pt"]:
    _p = CHECKPOINT_DIR / _f
    if _p.exists(): _p.unlink()
_r = RESULTS_DIR / "solar_sde_main_results.csv"
if _r.exists(): _r.unlink()
# Keep a copy of the closed-form per-horizon preds for the ensemble stage.
_pd_dir = RESULTS_DIR / "per_horizon_preds"
_cf_pd  = RESULTS_DIR / "closedform_preds"
if _pd_dir.exists():
    if _cf_pd.exists(): _sh.rmtree(_cf_pd)
    _sh.copytree(_pd_dir, _cf_pd); _sh.rmtree(_pd_dir)
    print(f"  stashed per-horizon preds -> {_cf_pd.name}/")
print("[OK] closed-form artifacts stashed; rollout STAGE 0 will train fresh.")
'''


STASH_ROLLOUT_CODE = '''\
# ==== Stash the rollout model's artifacts under rollout_* names ====
import shutil as _sh
_RO_CKPT = CHECKPOINT_DIR / "mdn_rollout_best.pt"
_RO_RES  = RESULTS_DIR / "rollout_main_results.csv"
if (CHECKPOINT_DIR / "mdn_v2_best.pt").exists():
    _sh.copy(CHECKPOINT_DIR / "mdn_v2_best.pt", _RO_CKPT)
    print(f"  stashed rollout ckpt -> {_RO_CKPT.name}")
if (RESULTS_DIR / "solar_sde_main_results.csv").exists():
    _sh.copy(RESULTS_DIR / "solar_sde_main_results.csv", _RO_RES)
    print(f"  stashed rollout results -> {_RO_RES.name}")
_pd_dir = RESULTS_DIR / "per_horizon_preds"
_ro_pd  = RESULTS_DIR / "rollout_preds"
if _pd_dir.exists():
    if _ro_pd.exists(): _sh.rmtree(_ro_pd)
    _sh.copytree(_pd_dir, _ro_pd)
    print(f"  stashed per-horizon preds -> {_ro_pd.name}/")
print("[OK] rollout artifacts stashed.")
'''


CHAMPION_SELECT_CODE = '''\
# ==== Champion selection (ON VALIDATION — legitimate model selection) ====
# Evaluates closed-form, rollout, and their ensemble on the VALIDATION split at
# the SkyGPT horizon (15 min), picks the lowest val CRPS, and restores that
# variant's artifacts as the canonical mdn_v2_best.pt / solar_sde_main_results.csv
# so every downstream stage (ablations, stratified, CV, economics, figures) runs
# on the champion. Test data is never touched for selection.
import shutil as _sh
SEQ = int(globals().get("SEQ_LEN", 16))
H_SEL = 15 if 15 in HORIZON_MIN.values() else sorted(HORIZON_MIN.values())[len(HORIZON_MIN)//2]
h_sel_steps = [k for k, v in HORIZON_MIN.items() if v == H_SEL][0]
print("=" * 70)
print(f"CHAMPION SELECTION on VALIDATION (h={H_SEL}min, CRPS)")
print("=" * 70)

_CF_CKPT = CHECKPOINT_DIR / "mdn_closedform_best.pt"
_RO_CKPT = CHECKPOINT_DIR / "mdn_rollout_best.pt"
have_cf = _CF_CKPT.exists(); have_ro = _RO_CKPT.exists()
print(f"  closed-form ckpt: {have_cf} | rollout ckpt: {have_ro}")
if not (have_cf or have_ro):
    raise RuntimeError("No trained model found — both STAGE 0 runs failed.")

def _load_model(kind):
    if kind == "closedform":
        m = ClosedFormSDE(z_dim=Z_DIM, c_dim=C_DIM, n_components=3, seq_len=SEQ,
                          d_model=128, n_heads=4, n_layers=2,
                          n_horizons=len(HORIZON_MIN)).to(DEVICE)
        m.load_state_dict(torch.load(_CF_CKPT, map_location=DEVICE, weights_only=False))
    else:
        m = RolloutLatentSDE(z_dim=Z_DIM, c_dim=C_DIM, n_components=3, seq_len=SEQ,
                             d_model=128, n_heads=4, n_layers=2,
                             n_horizons=len(HORIZON_MIN)).to(DEVICE)
        m.load_state_dict(torch.load(_RO_CKPT, map_location=DEVICE, weights_only=False))
    m.eval(); return m

def _val_samples(model, h, n_eval=1500, n_samp=50):
    """Per-point GHI/PV samples on the VALIDATION split at horizon h."""
    va = data["val"]
    idxs = list(range(SEQ - 1, min(SEQ - 1 + n_eval, len(va["Z"]) - h - 1)))
    yt_l, ys_l = [], []
    for k in range(0, len(idxs), 64):
        ch = idxs[k:k+64]; B = len(ch)
        z_seq = np.stack([va["Z"][i-SEQ+1:i+1] for i in ch]).astype(np.float32)
        kt_seq = np.stack([va["kt"][i-SEQ+1:i+1] for i in ch]).astype(np.float32)
        c_seq = (np.stack([va["cov"][i-SEQ+1:i+1] for i in ch]).astype(np.float32)
                 if va["cov"].shape[1] > 0 else np.zeros((B, SEQ, C_DIM), np.float32))
        cti = np.array([va["cti"][i] for i in ch], np.float32)[:, None]
        kt_t = np.array([va["kt"][i] for i in ch], np.float32)
        gcs_t = np.array([va["gcs"][i+h] for i in ch], np.float32)
        hn = np.full((B, 1), h/180.0, np.float32)
        with torch.no_grad():
            pi, mu, sd = model(torch.from_numpy(z_seq).to(DEVICE), torch.from_numpy(kt_seq).to(DEVICE),
                               torch.from_numpy(c_seq).to(DEVICE), torch.from_numpy(cti).to(DEVICE),
                               torch.from_numpy(hn).to(DEVICE))
            ds = mdn_sample(pi, mu, sd, n_samples=n_samp).cpu().numpy()
        ys_l.append(np.clip(kt_t[:, None] + ds, 0, 1.5) * gcs_t[:, None])
        yt_l.append(np.array([va["ghi"][i+h] for i in ch], np.float32))
    return np.concatenate(yt_l), np.concatenate(ys_l)

variants = {}
if have_cf:
    m_cf = _load_model("closedform")
    yt, ys_cf = _val_samples(m_cf, h_sel_steps)
    variants["closedform"] = float(crps_empirical(yt, ys_cf).mean())
    del m_cf; gc.collect()
if have_ro:
    m_ro = _load_model("rollout")
    yt, ys_ro = _val_samples(m_ro, h_sel_steps)
    variants["rollout"] = float(crps_empirical(yt, ys_ro).mean())
    del m_ro; gc.collect()
if have_cf and have_ro:
    # Ensemble: pool half the samples from each model (equal-weight mixture).
    n_half = ys_cf.shape[1] // 2
    ys_ens = np.concatenate([ys_cf[:, :n_half], ys_ro[:, :n_half]], axis=1)
    variants["ensemble"] = float(crps_empirical(yt, ys_ens).mean())
if torch.cuda.is_available(): torch.cuda.empty_cache()

for k, v in sorted(variants.items(), key=lambda t: t[1]):
    print(f"  val CRPS @ h={H_SEL}min  {k:12s} = {v:.4f}")
CHAMPION = min(variants, key=variants.get)
print(f"\\n  CHAMPION (selected on validation): {CHAMPION}")
pd.DataFrame([{"variant": k, "val_crps_h15": v, "champion": k == CHAMPION}
              for k, v in variants.items()]).to_csv(RESULTS_DIR / "champion_selection.csv", index=False)

# Restore the champion's artifacts as the canonical files for downstream stages.
# For the ensemble champion, downstream single-model stages (ablations etc.) run
# on the better single model, while the SkyGPT/benchmark tables report the ensemble.
_best_single = CHAMPION if CHAMPION != "ensemble" else min(
    {k: v for k, v in variants.items() if k != "ensemble"}, key=lambda k: variants[k])
_src_ckpt = _CF_CKPT if _best_single == "closedform" else _RO_CKPT
_src_res  = RESULTS_DIR / ("closedform_main_results.csv" if _best_single == "closedform" else "rollout_main_results.csv")
_src_pred = RESULTS_DIR / ("closedform_preds" if _best_single == "closedform" else "rollout_preds")
_sh.copy(_src_ckpt, CHECKPOINT_DIR / "mdn_v2_best.pt")
_sh.copy(_src_ckpt, CHECKPOINT_DIR / "sde_best.pt")
_sh.copy(_src_ckpt, CHECKPOINT_DIR / "score_best.pt")
if _src_res.exists(): _sh.copy(_src_res, RESULTS_DIR / "solar_sde_main_results.csv")
if _src_pred.exists():
    _dst = RESULTS_DIR / "per_horizon_preds"
    if _dst.exists(): _sh.rmtree(_dst)
    _sh.copytree(_src_pred, _dst)
    # refresh legacy npz for downstream consumers
    _h10 = 10 if 10 in HORIZON_MIN.values() else sorted(HORIZON_MIN.values())[0]
    _npz10 = _dst / f"solarsde_h{_h10}.npz"
    if _npz10.exists():
        _z = np.load(_npz10)
        np.savez(RESULTS_DIR / "test_predictions_h10min.npz",
                 y_true=_z["truths"], y_samples=_z["preds"],
                 is_ramp=_z["is_ramp"], truths=_z["truths"], preds=_z["preds"])
# Point the architecture alias at the champion's class so downstream stages
# (ablations, sampling, compute) instantiate the right model.
TemporalLatentSDE = ClosedFormSDE if _best_single == "closedform" else RolloutLatentSDE
CHAMPION_SINGLE = _best_single
print(f"  downstream suite runs on: {_best_single} (canonical mdn_v2_best.pt restored)")
'''


SKYGPT_TRIPLE_BENCHMARK_CODE = '''\
# ==== SkyGPT exact benchmark — ALL variants (closed-form, rollout, ensemble) ====
# Same protocol as the single-model SkyGPT stage: train on SKIPP'D benchmark
# release (2017-03..2019-10), test on SkyGPT's identical Nov-Dec 2019 cloudy
# file (zero leakage), CRPS/Winkler in kW, 90% PI, full 1-30 min band.
import h5py, datetime as _dt
import shutil as _sh
SKYGPT_DIR = DATA_DIR / "skygpt"; SKYGPT_DIR.mkdir(parents=True, exist_ok=True)
try:
    import gdown
except Exception:
    import subprocess as _sp; _sp.run([sys.executable, "-m", "pip", "install", "-q", "gdown"]); import gdown
for _fid, _nm in [("1VILdkCRWsDTrN9DPeMLh8jlibAoBLzy-", "test_set_2019nov_dec.hdf5"),
                  ("197pDAI8KVsiDAA1xaPbitZpmvzh9CDqT", "times_curr_test_2019nov_dec.npy")]:
    _d = SKYGPT_DIR / _nm
    if not (_d.exists() and _d.stat().st_size > 1000):
        print(f"  downloading {_nm} ..."); gdown.download(id=_fid, output=str(_d), quiet=True)

_hf = h5py.File(SKYGPT_DIR / "test_set_2019nov_dec.hdf5", "r")
imgs_log = _hf["test/images_log"][:]; pv_log = _hf["test/pv_log"][:].astype(np.float32)
pv_pred = _hf["test/pv_pred"][:].astype(np.float32)
sky_times = np.load(SKYGPT_DIR / "times_curr_test_2019nov_dec.npy", allow_pickle=True)
N_SKY = len(sky_times)
print("=" * 70)
print(f"SkyGPT EXACT BENCHMARK — all variants ({N_SKY} windows, 5 cloudy days)")
print("=" * 70)

# clear-sky lookup (reuse PREP's envelope if present)
if "SKIPPD_ENV" in globals():
    _ENV = SKIPPD_ENV
else:
    _lab = pd.concat([pd.read_parquet(DATA_DIR / "skippd" / "labels" / f"{s}-00000-of-00001.parquet")
                      for s in ["train", "test"]], ignore_index=True)
    _lab["time"] = pd.to_datetime(_lab["time"], utc=True).dt.tz_convert("US/Pacific")
    _lab["month"] = _lab["time"].dt.month; _lab["mod"] = _lab["time"].dt.hour * 60 + _lab["time"].dt.minute
    _ENV = (_lab.groupby(["month", "mod"])["pv"].quantile(0.92).rename("cs").reset_index()
            .sort_values(["month", "mod"]))
    _ENV["cs"] = _ENV.groupby("month")["cs"].transform(lambda s: s.rolling(31, center=True, min_periods=1).max())
_cs_lut = {(int(r.month), int(r.mod)): float(r.cs) for r in _ENV.itertuples()}
_cs_med = float(_ENV["cs"].median()); _cs_max = float(max(_ENV["cs"].max(), 1.0))
def _cs_at(dtm): return max(_cs_lut.get((dtm.month, dtm.hour * 60 + dtm.minute), _cs_med), 0.5)
_ctiscale = float(globals().get("_cti_scale", 1.0)) or 1.0

# VAE + motion features for log frames (identical to the training pipeline)
SEQ = 16
_vae = SkippdVAE(64).to(DEVICE)
_vae.load_state_dict(torch.load(CHECKPOINT_DIR / "skippd_vae.pt", map_location=DEVICE)); _vae.eval()
print("  encoding log frames ...")
_flat = imgs_log.reshape(-1, 64, 64, 3)
_Zlog = np.zeros((len(_flat), 64), np.float32)
with torch.no_grad():
    for k in range(0, len(_flat), 1024):
        xb = torch.from_numpy(_flat[k:k+1024].astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE) / 255.0
        mu, _ = _vae.encode(xb); _Zlog[k:k+len(mu)] = mu.cpu().numpy()
Zlog = _Zlog.reshape(N_SKY, 16, 64)
print("  computing motion features ...")
try:
    import cv2 as _cv2
except Exception:
    import subprocess as _sp; _sp.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"]); import cv2 as _cv2
_mn = CHECKPOINT_DIR / "motion_norm.npy"
_MMU, _MSD = (np.load(_mn) if _mn.exists() else (np.zeros(4, np.float32), np.ones(4, np.float32)))
_Hs = 64; _cyx = _Hs // 2; _r2s = (_Hs // 4) ** 2
_yy2, _xx2 = np.ogrid[:_Hs, :_Hs]; _sunm = ((_yy2 - _cyx) ** 2 + (_xx2 - _cyx) ** 2) <= _r2s
Mlog = np.zeros((N_SKY, 16, 4), np.float32)
for i in range(N_SKY):
    gs = [_cv2.cvtColor(imgs_log[i, j], _cv2.COLOR_RGB2GRAY) for j in range(16)]
    for j in range(1, 16):
        f = _cv2.calcOpticalFlowFarneback(gs[j-1], gs[j], None, 0.5, 3, 9, 3, 5, 1.2, 0)
        dx, dy = f[..., 0], f[..., 1]; mag = np.sqrt(dx*dx + dy*dy)
        Mlog[i, j] = [dx.mean(), dy.mean(), mag.mean(), mag[_sunm].mean()]
Mlog = ((Mlog - _MMU) / _MSD).astype(np.float32)

def _build_cov(ft, cs, motion):
    base = np.array([np.sin(2*np.pi*(ft.hour*60+ft.minute)/1440), np.cos(2*np.pi*(ft.hour*60+ft.minute)/1440),
                     np.sin(2*np.pi*ft.month/12), np.cos(2*np.pi*ft.month/12), cs/_cs_max], np.float32)
    base9 = np.concatenate([base, motion]).astype(np.float32)
    v = np.zeros(C_DIM, np.float32); n = len(base9)
    v[:min(n, C_DIM)] = base9[:min(n, C_DIM)]
    if C_DIM >= 2 * n: v[n:2*n] = base9
    return v

# continuous PV series -> targets at any horizon
_series = {}
for i in range(N_SKY):
    tc = sky_times[i]
    for j in range(16): _series[tc - _dt.timedelta(minutes=15 - j)] = float(pv_log[i, j])
    for j in range(15): _series[tc + _dt.timedelta(minutes=j + 1)] = float(pv_pred[i, j])

# load whichever models exist
_models = {}
_CF = CHECKPOINT_DIR / "mdn_closedform_best.pt"; _RO = CHECKPOINT_DIR / "mdn_rollout_best.pt"
if _CF.exists():
    m = ClosedFormSDE(z_dim=Z_DIM, c_dim=C_DIM, n_components=3, seq_len=SEQ, d_model=128,
                      n_heads=4, n_layers=2, n_horizons=len(HORIZON_MIN)).to(DEVICE)
    m.load_state_dict(torch.load(_CF, map_location=DEVICE, weights_only=False)); m.eval()
    _models["closedform"] = m
if _RO.exists():
    m = RolloutLatentSDE(z_dim=Z_DIM, c_dim=C_DIM, n_components=3, seq_len=SEQ, d_model=128,
                         n_heads=4, n_layers=2, n_horizons=len(HORIZON_MIN)).to(DEVICE)
    m.load_state_dict(torch.load(_RO, map_location=DEVICE, weights_only=False)); m.eval()
    _models["rollout"] = m
print(f"  variants available: {list(_models.keys())}"
      + (" + ensemble" if len(_models) == 2 else ""))

rng_sky = np.random.RandomState(0)
SKY_H = [h for h in HORIZONS]
all_rows = []
for h in SKY_H:
    valid = [i for i in range(N_SKY) if (sky_times[i] + _dt.timedelta(minutes=h)) in _series]
    if len(valid) < 50: continue
    # build batch inputs once per horizon
    samp_by_variant = {k: [] for k in _models}
    yt_l, sp_l = [], []
    for k0 in range(0, len(valid), 256):
        idx = valid[k0:k0 + 256]; B = len(idx)
        zb = np.stack([Zlog[i] for i in idx]).astype(np.float32)
        ktb = np.zeros((B, 16), np.float32); covb = np.zeros((B, 16, C_DIM), np.float32)
        ctib = np.zeros(B, np.float32); kt_t = np.zeros(B, np.float32)
        cs_tph = np.zeros(B, np.float32); tgt = np.zeros(B, np.float32)
        for bi, i in enumerate(idx):
            tc = sky_times[i]
            for j in range(16):
                ft = tc - _dt.timedelta(minutes=(15 - j)); cs = _cs_at(ft)
                ktb[bi, j] = min(pv_log[i, j] / cs, 1.3)
                covb[bi, j] = _build_cov(ft, cs, Mlog[i, j])
            v = np.diff(zb[bi][6:], axis=0)
            ctib[bi] = min(np.linalg.norm(np.var(v, axis=0)) / _ctiscale, 10.0)
            kt_t[bi] = ktb[bi, -1]
            cs_tph[bi] = _cs_at(tc + _dt.timedelta(minutes=h))
            tgt[bi] = _series[tc + _dt.timedelta(minutes=h)]
        hn = np.full((B, 1), h / 180.0, np.float32)
        tens = [torch.from_numpy(a).to(DEVICE) for a in (zb, ktb, covb, ctib[:, None], hn)]
        for name, m in _models.items():
            with torch.no_grad():
                pi, mu, sd = m(*tens)
                ds = mdn_sample(pi, mu, sd, n_samples=N_SAMPLES).cpu().numpy()
            samp_by_variant[name].append(np.clip(kt_t[:, None] + ds, 0, 1.5) * cs_tph[:, None])
        sp_mean = kt_t * cs_tph
        sp_sig = max(float(np.std(tgt - sp_mean)), 1e-3)
        sp_l.append(np.clip(sp_mean[:, None] + rng_sky.randn(B, N_SAMPLES) * sp_sig, 0, None))
        yt_l.append(tgt)
    yt = np.concatenate(yt_l); sp = np.concatenate(sp_l)
    sp_crps = float(crps_empirical(yt, sp).mean())
    samp_full = {k: np.concatenate(v) for k, v in samp_by_variant.items()}
    if len(samp_full) == 2:
        nh = N_SAMPLES // 2
        samp_full["ensemble"] = np.concatenate(
            [samp_full["closedform"][:, :nh], samp_full["rollout"][:, :nh]], axis=1)
    for name, ys in samp_full.items():
        crps = float(crps_empirical(yt, ys).mean())
        wink = winkler_score(yt, ys, 0.9); picp = picp_metric(yt, ys, 0.9)
        skill = 100.0 * (sp_crps - crps) / max(sp_crps, 1e-9)
        all_rows.append({"variant": name, "horizon_min": h, "crps_kW": round(crps, 3),
                         "winkler": round(wink, 2), "picp": round(picp, 3),
                         "smart_pers_crps": round(sp_crps, 3),
                         "skill_vs_smartpers_%": round(skill, 1), "n_eval": len(yt)})
    best = min(samp_full, key=lambda k: float(crps_empirical(yt, samp_full[k]).mean()))
    print(f"  h={h:2d}min  " + "  ".join(
        f"{k}={float(crps_empirical(yt, v).mean()):.3f}" for k, v in samp_full.items())
        + f"  smart-pers={sp_crps:.3f}  [best: {best}]")

sky_all = pd.DataFrame(all_rows)
sky_all.to_csv(RESULTS_DIR / "skygpt_benchmark_all_variants.csv", index=False)

# Head-to-head table at h=15 with published numbers
_pub_rows = []
for name in sky_all["variant"].unique():
    _r = sky_all[(sky_all.variant == name) & (sky_all.horizon_min == 15)]
    if len(_r):
        _pub_rows.append({"method": f"SolarSDE-{name} (ours)",
                          "crps_kW": float(_r["crps_kW"].iloc[0]),
                          "winkler": float(_r["winkler"].iloc[0]),
                          "skill_vs_smartpers_%": float(_r["skill_vs_smartpers_%"].iloc[0])})
_pub_rows += [
    {"method": "SkyGPT->U-Net (published)", "crps_kW": 2.81, "winkler": 26.70, "skill_vs_smartpers_%": 23.0},
    {"method": "SUNSET (published)",        "crps_kW": 3.31, "winkler": 56.95, "skill_vs_smartpers_%": 9.8},
    {"method": "smart persistence (published)", "crps_kW": 3.67, "winkler": float("nan"), "skill_vs_smartpers_%": 0.0},
]
head = pd.DataFrame(_pub_rows).sort_values("crps_kW")
head.to_csv(RESULTS_DIR / "skygpt_headline_h15_all.csv", index=False)
print("\\nHEAD-TO-HEAD at h=15min (SkyGPT's identical cloudy test set):")
print(head.to_string(index=False))
_ours = head[head.method.str.contains("ours")]
_best_ours = _ours.iloc[0] if len(_ours) else None
if _best_ours is not None:
    if _best_ours.crps_kW < 2.81:
        print(f"\\n  >>> {_best_ours.method} BEATS SkyGPT: {_best_ours.crps_kW:.3f} < 2.81 kW <<<")
    else:
        _gap = 100 * (_best_ours.crps_kW - 2.81) / 2.81
        print(f"\\n  best ours = {_best_ours.crps_kW:.3f} kW vs SkyGPT 2.81 ({_gap:+.1f}%) — "
              f"report honestly; the multi-horizon/calibration contributions stand regardless.")
print("  -> saved skygpt_benchmark_all_variants.csv, skygpt_headline_h15_all.csv")
for m in _models.values(): del m
del _vae; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
'''
