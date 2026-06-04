"""SkyGPT exact-benchmark evaluation for the SKIPP'D notebook.

Evaluates the trained SolarSDE on SkyGPT's *identical* held-out test set
(test_set_2019nov_dec.hdf5: 5 cloudy days, Nov-Dec 2019, 2,582 windows, 30-kW
system) and reports CRPS + Winkler + PICP at h=1,5,10,15, side-by-side with the
published SkyGPT numbers. This is the head-to-head a SKIPP'D-familiar reviewer
asks for.

Published reference (Nie et al. 2024, Adv. Applied Energy; arXiv 2306.11682),
15-min horizon on this exact test set:
    SkyGPT->U-Net     CRPS 2.81 kW   Winkler 26.70
    SUNSET            CRPS 3.31 kW   Winkler 56.95
    smart persistence CRPS 3.67 kW
SkyGPT's headline claim: +23% CRPS over smart persistence, +13% over SUNSET.

Protocol note (disclosed): we TRAIN on the SKIPP'D benchmark release
(2017-03..2019-10, same site/period as SkyGPT's train) and TEST on SkyGPT's
identical Nov-Dec 2019 file — zero leakage (train ends Oct 26, test starts
Nov 12). Each test window provides 16 one-minute log frames (matching SEQ_LEN=16)
and 15 future PV values (pv_pred[:,h-1] = PV at t+h min), so horizons 1..15 are
directly available on the identical set.
"""

SKYGPT_BENCHMARK_CODE = '''\
# ==== SkyGPT exact-test benchmark (identical Nov-Dec 2019 cloudy test set) ====
import h5py, datetime as _dt
SKYGPT_DIR = DATA_DIR / "skygpt"; SKYGPT_DIR.mkdir(parents=True, exist_ok=True)
try:
    import gdown
except Exception:
    import subprocess as _sp; _sp.run([sys.executable, "-m", "pip", "install", "-q", "gdown"]); import gdown

_GD = [("1VILdkCRWsDTrN9DPeMLh8jlibAoBLzy-", "test_set_2019nov_dec.hdf5"),
       ("197pDAI8KVsiDAA1xaPbitZpmvzh9CDqT", "times_curr_test_2019nov_dec.npy")]
for _fid, _nm in _GD:
    _d = SKYGPT_DIR / _nm
    if not (_d.exists() and _d.stat().st_size > 1000):
        print(f"  downloading {_nm} ...", flush=True)
        gdown.download(id=_fid, output=str(_d), quiet=True)
if not (SKYGPT_DIR / "test_set_2019nov_dec.hdf5").exists():
    raise RuntimeError("SkyGPT test set download failed — re-run this cell.")

_hf = h5py.File(SKYGPT_DIR / "test_set_2019nov_dec.hdf5", "r")
imgs_log = _hf["test/images_log"][:]          # (N,16,64,64,3) uint8
pv_log   = _hf["test/pv_log"][:].astype(np.float32)    # (N,16)
pv_pred  = _hf["test/pv_pred"][:].astype(np.float32)   # (N,15) -> t+1..t+15
sky_times = np.load(SKYGPT_DIR / "times_curr_test_2019nov_dec.npy", allow_pickle=True)
N_SKY = len(sky_times)
_days = sorted(set(t.date() for t in sky_times))
print("=" * 70)
print(f"SkyGPT EXACT test: {N_SKY} windows, {len(_days)} cloudy days {[str(d) for d in _days]}")
print("=" * 70)

# --- clear-sky-PV envelope (reuse PREP's if present, else rebuild from labels) ---
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
def _cs_at(dt):
    return max(_cs_lut.get((dt.month, dt.hour * 60 + dt.minute), _cs_med), 0.5)
_ctiscale = float(globals().get("_cti_scale", 1.0)) or 1.0

# --- trained VAE + SDE ---
SEQ = 16
_vae = SkippdVAE(64).to(DEVICE)
_vae.load_state_dict(torch.load(CHECKPOINT_DIR / "skippd_vae.pt", map_location=DEVICE)); _vae.eval()
# use the arch factory if present (notebook 12 arch variants), else default dims
if "build_sde_model" in globals():
    _sde = build_sde_model(SEQ)
else:
    _sde = TemporalLatentSDE(z_dim=Z_DIM, c_dim=C_DIM, n_components=3, seq_len=SEQ,
                             d_model=128, n_heads=4, n_layers=2,
                             n_horizons=len(HORIZON_MIN)).to(DEVICE)
_sde.load_state_dict(torch.load(CHECKPOINT_DIR / "mdn_v2_best.pt", map_location=DEVICE)); _sde.eval()

# --- encode all 16 log frames per window -> Zlog (N,16,64) ---
print("  encoding log frames ...")
_flat = imgs_log.reshape(-1, 64, 64, 3)
_Zlog = np.zeros((len(_flat), 64), np.float32)
with torch.no_grad():
    for k in range(0, len(_flat), 1024):
        xb = torch.from_numpy(_flat[k:k+1024].astype(np.float32)).permute(0, 3, 1, 2).to(DEVICE) / 255.0
        mu, _ = _vae.encode(xb); _Zlog[k:k+len(mu)] = mu.cpu().numpy()
Zlog = _Zlog.reshape(N_SKY, 16, 64)

# --- optical-flow motion features per log frame (must match training pipeline) ---
print("  computing motion features for log frames ...")
try:
    import cv2 as _cv2
except Exception:
    import subprocess as _sp; _sp.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"]); import cv2 as _cv2
_mn_path = CHECKPOINT_DIR / "motion_norm.npy"
if _mn_path.exists():
    _MMU, _MSD = np.load(_mn_path)
else:
    _MMU, _MSD = np.zeros(4, np.float32), np.ones(4, np.float32)
_Hs = 64; _cyx = _Hs // 2; _r2s = (_Hs // 4) ** 2
_yy2, _xx2 = np.ogrid[:_Hs, :_Hs]; _sunm = ((_yy2 - _cyx) ** 2 + (_xx2 - _cyx) ** 2) <= _r2s
Mlog = np.zeros((N_SKY, 16, 4), np.float32)
for i in range(N_SKY):
    gs = [_cv2.cvtColor(imgs_log[i, j], _cv2.COLOR_RGB2GRAY) for j in range(16)]
    for j in range(1, 16):
        f = _cv2.calcOpticalFlowFarneback(gs[j-1], gs[j], None, 0.5, 3, 9, 3, 5, 1.2, 0)
        dx, dy = f[..., 0], f[..., 1]; mag = np.sqrt(dx*dx + dy*dy)
        Mlog[i, j] = [dx.mean(), dy.mean(), mag.mean(), mag[_sunm].mean()]
Mlog = ((Mlog - _MMU) / _MSD).astype(np.float32)   # normalize with training stats

def _build_cov(ft, cs, motion):
    # match LOAD_DATA layout: [base9 = time/sky(5)+motion(4)] duplicated, then image zeros
    base = np.array([np.sin(2*np.pi*(ft.hour*60+ft.minute)/1440), np.cos(2*np.pi*(ft.hour*60+ft.minute)/1440),
                     np.sin(2*np.pi*ft.month/12), np.cos(2*np.pi*ft.month/12), cs/_cs_max], np.float32)
    base9 = np.concatenate([base, motion]).astype(np.float32)   # 9 dims
    v = np.zeros(C_DIM, np.float32)
    n = len(base9)
    v[:min(n, C_DIM)] = base9[:min(n, C_DIM)]
    if C_DIM >= 2 * n: v[n:2*n] = base9                          # physics dup
    return v

# Reconstruct a continuous 1-min PV series for the 5 cloudy days from the
# overlapping windows (pv_log = t-15..t observed, pv_pred = t+1..t+15). This lets
# us read targets at ANY horizon — so we evaluate the FULL 1-30 min band on
# SkyGPT's identical cloudy test days, not just <=15. (h=15 is the SkyGPT
# head-to-head; 1/5/10 are uncontested short nowcasts; 20/30 extend beyond
# SkyGPT entirely on the same hard cloudy data.)
_series = {}
for i in range(N_SKY):
    tc = sky_times[i]
    for j in range(16): _series[tc - _dt.timedelta(minutes=15 - j)] = float(pv_log[i, j])
    for j in range(15): _series[tc + _dt.timedelta(minutes=j + 1)] = float(pv_pred[i, j])

SKY_H = list(HORIZONS)                       # full band on the exact cloudy set
rng_sky = np.random.RandomState(0)
rows = []
for h in SKY_H:
    # windows whose t+h target exists in the reconstructed series
    valid = [i for i in range(N_SKY) if (sky_times[i] + _dt.timedelta(minutes=h)) in _series]
    if len(valid) < 50:
        continue
    yt_l, ys_l, sp_l = [], [], []
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
            v = np.diff(zb[bi][6:], axis=0)                      # last ~10 frames
            ctib[bi] = min(np.linalg.norm(np.var(v, axis=0)) / _ctiscale, 10.0)
            kt_t[bi] = ktb[bi, -1]
            cs_tph[bi] = _cs_at(tc + _dt.timedelta(minutes=h))
            tgt[bi] = _series[tc + _dt.timedelta(minutes=h)]
        hn = np.full((B, 1), h / 180.0, np.float32)
        with torch.no_grad():
            pi, mu, sd = _sde(torch.from_numpy(zb).to(DEVICE), torch.from_numpy(ktb).to(DEVICE),
                              torch.from_numpy(covb).to(DEVICE), torch.from_numpy(ctib[:, None]).to(DEVICE),
                              torch.from_numpy(hn).to(DEVICE))
            ds = mdn_sample(pi, mu, sd, n_samples=N_SAMPLES).cpu().numpy()
        pv_s = np.clip(kt_t[:, None] + ds, 0, 1.5) * cs_tph[:, None]
        # smart persistence: kt persists, x clear-sky at t+h, with kt-residual noise
        sp_mean = kt_t * cs_tph
        sp_sig = max(float(np.std(tgt - sp_mean)), 1e-3)
        sp_s = np.clip(sp_mean[:, None] + rng_sky.randn(B, N_SAMPLES) * sp_sig, 0, None)
        yt_l.append(tgt); ys_l.append(pv_s); sp_l.append(sp_s)
    yt = np.concatenate(yt_l); ys = np.concatenate(ys_l); sp = np.concatenate(sp_l)
    crps = float(crps_empirical(yt, ys).mean()); wink = winkler_score(yt, ys, 0.9); picp = picp_metric(yt, ys, 0.9)
    sp_crps = float(crps_empirical(yt, sp).mean())
    skill = 100.0 * (sp_crps - crps) / max(sp_crps, 1e-9)
    rows.append({"horizon_min": h, "crps_kW": round(crps, 3), "winkler": round(wink, 2),
                 "picp": round(picp, 3), "smart_pers_crps": round(sp_crps, 3),
                 "skill_vs_smartpers_%": round(skill, 1), "n_eval": len(yt)})
    print(f"  h={h:2d}min  CRPS={crps:.3f} kW  Winkler={wink:.2f}  PICP={picp:.3f}  "
          f"smart-pers CRPS={sp_crps:.3f}  skill={skill:+.1f}%  (n={len(yt)})")

sky_df = pd.DataFrame(rows)
sky_df.to_csv(RESULTS_DIR / "skygpt_benchmark_comparison.csv", index=False)

# --- head-to-head table at h=15 (SkyGPT's horizon) ---
_pub = pd.DataFrame([
    {"method": "SolarSDE (ours)",  "crps_kW": float(sky_df.loc[sky_df.horizon_min == 15, "crps_kW"].iloc[0]) if (sky_df.horizon_min == 15).any() else float("nan"),
     "winkler": float(sky_df.loc[sky_df.horizon_min == 15, "winkler"].iloc[0]) if (sky_df.horizon_min == 15).any() else float("nan"),
     "skill_vs_smartpers_%": float(sky_df.loc[sky_df.horizon_min == 15, "skill_vs_smartpers_%"].iloc[0]) if (sky_df.horizon_min == 15).any() else float("nan")},
    {"method": "SkyGPT->U-Net (pub)", "crps_kW": 2.81, "winkler": 26.70, "skill_vs_smartpers_%": 23.0},
    {"method": "SUNSET (pub)",        "crps_kW": 3.31, "winkler": 56.95, "skill_vs_smartpers_%": 9.8},
    {"method": "smart persistence (pub)", "crps_kW": 3.67, "winkler": float("nan"), "skill_vs_smartpers_%": 0.0},
])
_pub.to_csv(RESULTS_DIR / "skygpt_headline_h15.csv", index=False)
print("\\nHEAD-TO-HEAD at h=15min (SkyGPT's identical test set):")
print(_pub.to_string(index=False))
print("\\n  Multi-horizon (ours, same exact test set):")
print(sky_df.to_string(index=False))
print("  -> saved skygpt_benchmark_comparison.csv, skygpt_headline_h15.csv")
print("  [protocol] trained on SKIPP'D 2017-03..2019-10; tested on SkyGPT's identical "
      "Nov-Dec 2019 cloudy file (no leakage). CRPS/Winkler in kW, 90% PI.")
del _vae, _sde; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()
'''
