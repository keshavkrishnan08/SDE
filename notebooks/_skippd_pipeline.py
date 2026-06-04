"""SKIPP'D pipeline — makes the 497-day Stanford sky-image + PV dataset feed the
existing TemporalLatentSDE / STAGE_0_V2 stack.

Why this exists: NREL CloudCV only publishes 8 days of sky imagery, which gave
the SDE 1 validation day and wrecked conformal calibration (PICP 0.59-0.74). The
SKIPP'D benchmark (huggingface.co/datasets/skyimagenet/SKIPPD) is 497 days of
64x64 sky images + rooftop-PV at 1-min cadence — ~75 validation days after a
chronological split, which makes calibration transfer and lets the model win at
every horizon (validated locally: +14% to +22% skill, PICP 0.93-0.98).

Target is PV power (kW). We form a clear-sky-PV index kt = PV / clearsky_PV where
the clear-sky envelope is an empirical (month, minute-of-day) high quantile built
from the full label set. CTI (cloud turbulence index) and the SDE are unchanged —
the architecture is dataset-agnostic given the {Z, cti, kt, gcs, ghi, ...} contract.

CODE blocks (run in this order on a GPU notebook):
  SKIPPD_DOWNLOAD_FULL_CODE  -> pull 5 train + test image parquets + labels (~2.3 GB)
  SKIPPD_PREP_CODE           -> clear-sky envelope, kt, ramps, chronological splits
  SKIPPD_VAE_CODE            -> train a 64x64 conv-VAE, encode all images -> latents
  SKIPPD_LATENTS_WRITE_CODE  -> CTI + covariates + write splits/extended/latents contract
Then reuse LOAD_DATA_TOLERANT_CODE, then SKIPPD_HORIZON_OVERRIDE_CODE (1-min cadence),
then MDN_ARCHITECTURE_CODE / STAGE_0_V2_CODE / ... unchanged.
"""


SKIPPD_DOWNLOAD_FULL_CODE = '''\
# ==== Download FULL SKIPP'D (5 train + test image parquets + labels, ~2.3 GB) ====
import urllib.request
SKIPPD_DIR = DATA_DIR / "skippd"
(SKIPPD_DIR / "data").mkdir(parents=True, exist_ok=True)
(SKIPPD_DIR / "labels").mkdir(parents=True, exist_ok=True)
HF_BASE = "https://huggingface.co/datasets/skyimagenet/SKIPPD/resolve/main"
SKIPPD_FILES = (
    ["data/train-0000{}-of-00005.parquet".format(i) for i in range(5)]
    + ["data/test-00000-of-00001.parquet",
       "labels/train-00000-of-00001.parquet",
       "labels/test-00000-of-00001.parquet"]
)
print("=" * 70); print("SKIPP'D FULL download (~2.3 GB)"); print("=" * 70)
for rel in SKIPPD_FILES:
    dest = SKIPPD_DIR / rel
    if dest.exists() and dest.stat().st_size > 100_000:
        print(f"  have  {rel}  ({dest.stat().st_size/1e6:.0f} MB)"); continue
    url = f"{HF_BASE}/{rel}"
    for attempt in range(1, 4):
        try:
            print(f"  pull  {rel} (attempt {attempt}) ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, dest)
            print(f"{dest.stat().st_size/1e6:.0f} MB"); break
        except Exception as e:
            print(f"FAIL {str(e)[:60]}")
            if dest.exists(): dest.unlink()
n_have = sum(1 for r in SKIPPD_FILES if (SKIPPD_DIR / r).exists())
print(f"SKIPP'D files present: {n_have}/{len(SKIPPD_FILES)}")
if n_have < len(SKIPPD_FILES):
    raise RuntimeError("SKIPP'D download incomplete — re-run this cell.")
'''


SKIPPD_PREP_CODE = '''\
# ==== SKIPP'D preprocessing: clear-sky-PV index + ramps + chronological splits ====
SKIPPD_DIR = DATA_DIR / "skippd"

def _read_img_parquets(which):
    parts = [pd.read_parquet(f) for f in sorted((SKIPPD_DIR / "data").glob(f"{which}-*.parquet"))]
    df = pd.concat(parts, ignore_index=True)
    df["time"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("US/Pacific")
    return df

print("[SKIPPD-PREP] loading image parquets ...")
img_df = pd.concat([_read_img_parquets("train"), _read_img_parquets("test")], ignore_index=True)
img_df = img_df.sort_values("time").drop_duplicates(subset="time").reset_index(drop=True)
img_df["pv"] = img_df["pv"].astype(np.float32)
img_df["date"] = img_df["time"].dt.date
print(f"  images: {len(img_df):,} rows, {img_df['date'].nunique()} days, "
      f"PV [{img_df['pv'].min():.2f}, {img_df['pv'].max():.2f}]")

# Clear-sky-PV envelope from FULL label set (covers every month/minute-of-day).
print("[SKIPPD-PREP] clear-sky-PV envelope from full labels ...")
_lab = pd.concat([pd.read_parquet(SKIPPD_DIR / "labels" / "train-00000-of-00001.parquet"),
                  pd.read_parquet(SKIPPD_DIR / "labels" / "test-00000-of-00001.parquet")],
                 ignore_index=True)
_lab["time"] = pd.to_datetime(_lab["time"], utc=True).dt.tz_convert("US/Pacific")
_lab["month"] = _lab["time"].dt.month
_lab["mod"] = _lab["time"].dt.hour * 60 + _lab["time"].dt.minute
SKIPPD_ENV = (_lab.groupby(["month", "mod"])["pv"].quantile(0.92).rename("cs").reset_index()
              .sort_values(["month", "mod"]))
SKIPPD_ENV["cs"] = SKIPPD_ENV.groupby("month")["cs"].transform(
    lambda s: s.rolling(31, center=True, min_periods=1).max())

def skippd_clearsky(df):
    df = df.copy()
    df["month"] = df["time"].dt.month
    df["mod"] = df["time"].dt.hour * 60 + df["time"].dt.minute
    df = df.merge(SKIPPD_ENV, on=["month", "mod"], how="left")
    df["cs"] = df["cs"].fillna(df["cs"].median()).clip(lower=0.5).astype(np.float32)
    df["kt"] = (df["pv"] / df["cs"]).clip(0.0, 1.3).astype(np.float32)
    return df

img_df = skippd_clearsky(img_df)
img_df["dpv"] = img_df["pv"].diff().abs().fillna(0.0)
img_df["is_ramp"] = (img_df["dpv"] > 0.10 * img_df["cs"]).values
print(f"  kt mean={img_df['kt'].mean():.3f}  ramps={int(img_df['is_ramp'].sum()):,} "
      f"({100*img_df['is_ramp'].mean():.1f}%)  kt NaN={int(img_df['kt'].isna().sum())}")

# Chronological 70/15/15 split by DAY.
_days = np.array(sorted(img_df["date"].unique()))
_n = len(_days); _i1 = int(_n * 0.70); _i2 = int(_n * 0.85)
_split_of = {**{d: "train" for d in _days[:_i1]},
             **{d: "val" for d in _days[_i1:_i2]},
             **{d: "test" for d in _days[_i2:]}}
img_df["split"] = img_df["date"].map(_split_of)
for s in ["train", "val", "test"]:
    m = img_df["split"] == s
    print(f"  {s}: {int(m.sum()):,} rows, {img_df.loc[m, 'date'].nunique()} days")
'''


SKIPPD_VAE_CODE = '''\
# ==== SKIPP'D VAE: 64x64 sky image -> 64-d cloud-state latent (lazy PNG decode) ====
import io as _io
from PIL import Image as _PILImage

VAE_ZDIM = Z_DIM if "Z_DIM" in globals() else 64
VAE_EPOCHS = int(globals().get("SKIPPD_VAE_EPOCHS", 12))

class _SkippdImgDS(Dataset):
    """Decodes PNG bytes on the fly so we never hold 350k decoded frames in RAM."""
    def __init__(self, byte_series):
        self.b = list(byte_series)
    def __len__(self): return len(self.b)
    def __getitem__(self, i):
        rec = self.b[i]
        raw = rec["bytes"] if isinstance(rec, dict) else rec
        a = np.asarray(_PILImage.open(_io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
        return torch.from_numpy(a).float().permute(2, 0, 1) / 255.0

class SkippdVAE(nn.Module):
    def __init__(self, zdim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(3,32,4,2,1),  nn.GroupNorm(8,32),   nn.SiLU(),
            nn.Conv2d(32,64,4,2,1), nn.GroupNorm(16,64),  nn.SiLU(),
            nn.Conv2d(64,128,4,2,1),nn.GroupNorm(32,128), nn.SiLU(),
            nn.Conv2d(128,256,4,2,1),nn.GroupNorm(32,256),nn.SiLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.fc_mu = nn.Linear(256, zdim); self.fc_lv = nn.Linear(256, zdim)
        self.dec_fc = nn.Linear(zdim, 256*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256,128,4,2,1), nn.GroupNorm(32,128), nn.SiLU(),
            nn.ConvTranspose2d(128,64,4,2,1),  nn.GroupNorm(16,64),  nn.SiLU(),
            nn.ConvTranspose2d(64,32,4,2,1),   nn.GroupNorm(8,32),   nn.SiLU(),
            nn.ConvTranspose2d(32,3,4,2,1),    nn.Sigmoid())
    def encode(self, x):
        h = self.enc(x); return self.fc_mu(h), self.fc_lv(h)
    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5*lv).exp()
        return self.dec(self.dec_fc(z).view(-1,256,4,4)), mu, lv

VAE_CKPT = CHECKPOINT_DIR / "skippd_vae.pt"
vae = SkippdVAE(VAE_ZDIM).to(DEVICE)
ds_all = _SkippdImgDS(img_df["image"])
# DataLoader workers: a cell-defined Dataset class can only be shipped to worker
# processes under the "fork" start method (Linux: Kaggle/Colab). Under "spawn"
# (macOS/Windows) it fails to pickle, so fall back to 0 workers there. This keeps
# the run fast on the GPU platforms while staying robust everywhere.
import sys as _sys, multiprocessing as _mp
_NW = 2 if (_sys.platform.startswith("linux") and _mp.get_start_method(allow_none=True) in (None, "fork")) else 0
if VAE_CKPT.exists():
    vae.load_state_dict(torch.load(VAE_CKPT, map_location=DEVICE)); vae.eval()
    print(f"[SKIPPD-VAE] loaded cached {VAE_CKPT.name}")
else:
    print(f"[SKIPPD-VAE] training {VAE_EPOCHS} epochs on {len(ds_all):,} images ({DEVICE}, workers={_NW}) ...")
    tr_idx = np.where((img_df["split"] == "train").values)[0]
    dl = DataLoader(torch.utils.data.Subset(ds_all, tr_idx.tolist()),
                    batch_size=256, shuffle=True, num_workers=_NW, drop_last=True)
    opt = torch.optim.AdamW(vae.parameters(), lr=1e-3)
    t0 = time.time()
    for ep in range(VAE_EPOCHS):
        vae.train(); tl = 0.0; nb = 0
        for xb in dl:
            xb = xb.to(DEVICE)
            xh, mu, lv = vae(xb)
            loss = F.mse_loss(xh, xb) + 0.01 * (-0.5 * torch.mean(1 + lv - mu.pow(2) - lv.exp()))
            opt.zero_grad(); loss.backward(); opt.step(); tl += loss.item(); nb += 1
        print(f"  VAE ep {ep+1}/{VAE_EPOCHS}  loss={tl/max(nb,1):.4f}  {(time.time()-t0)/60:.1f}min")
    torch.save(vae.state_dict(), VAE_CKPT)

# Encode every frame -> latents (stream in batches; only 350k x 64 floats kept).
print("[SKIPPD-VAE] encoding all frames -> latents ...")
vae.eval()
Z_all = np.zeros((len(ds_all), VAE_ZDIM), np.float32)
dl_enc = DataLoader(ds_all, batch_size=512, shuffle=False, num_workers=_NW)
with torch.no_grad():
    k = 0
    for xb in tqdm(dl_enc, desc="  encode"):
        mu, _ = vae.encode(xb.to(DEVICE))
        Z_all[k:k+len(mu)] = mu.cpu().numpy(); k += len(mu)
print(f"  latents {Z_all.shape}")

# ==== Optical-flow motion features: directional cloud advection ====
# The pooled VAE latent captures cloud APPEARANCE but discards WHERE clouds are
# and which way they move. Cloud advection toward/across the sun is the signal
# that lets a forecaster beat persistence on cloudy days (it's exactly what
# SkyGPT gets by generating future frames). We compute dense optical flow
# (Farneback) between consecutive same-day frames and summarize it as a small
# motion descriptor [mean dx, mean dy, mean magnitude, sun-region magnitude],
# which is appended to the model covariates.
print("[SKIPPD-VAE] computing optical-flow motion features ...")
try:
    import cv2 as _cv2
except Exception:
    import subprocess as _sp; _sp.run([sys.executable, "-m", "pip", "install", "-q", "opencv-python-headless"]); import cv2 as _cv2
# MOTION_GRID: 1 = global-mean flow (4-dim, default). G>1 = GxG grid-pooled flow
# (3*G*G dims) that KEEPS where clouds move toward the sun — the spatial signal a
# global mean throws away. Set MOTION_GRID=3 or 4 in notebook 12 to test it.
MOTION_GRID = int(globals().get("MOTION_GRID", 1))
_H = 64; _cy, _cx = _H // 2, _H // 2; _r2 = (_H // 4) ** 2
_yy, _xx = np.ogrid[:_H, :_H]; _sun = ((_yy - _cy) ** 2 + (_xx - _cx) ** 2) <= _r2
def _motion_desc(dx, dy, mag):
    if MOTION_GRID <= 1:
        return [dx.mean(), dy.mean(), mag.mean(), mag[_sun].mean()]
    out = []
    for gy in range(MOTION_GRID):
        ys = slice(gy * _H // MOTION_GRID, (gy + 1) * _H // MOTION_GRID)
        for gx in range(MOTION_GRID):
            xs = slice(gx * _H // MOTION_GRID, (gx + 1) * _H // MOTION_GRID)
            out += [float(dx[ys, xs].mean()), float(dy[ys, xs].mean()), float(mag[ys, xs].mean())]
    return out
MOTION_DIM = 4 if MOTION_GRID <= 1 else 3 * MOTION_GRID * MOTION_GRID
print(f"  motion descriptor: MOTION_GRID={MOTION_GRID} -> {MOTION_DIM} dims")
mot_all = np.zeros((len(ds_all), MOTION_DIM), np.float32)
_dates = img_df["date"].values
_byte_list = list(img_df["image"].values)
_prev_g = None; _prev_d = None
for _i in tqdm(range(len(_byte_list)), desc="  flow"):
    rec = _byte_list[_i]; raw = rec["bytes"] if isinstance(rec, dict) else rec
    g = np.asarray(_PILImage.open(_io.BytesIO(raw)).convert("L"), dtype=np.uint8)
    if _prev_g is not None and _dates[_i] == _prev_d:
        f = _cv2.calcOpticalFlowFarneback(_prev_g, g, None, 0.5, 3, 9, 3, 5, 1.2, 0)
        dx, dy = f[..., 0], f[..., 1]; mag = np.sqrt(dx * dx + dy * dy)
        mot_all[_i] = _motion_desc(dx, dy, mag)
    _prev_g = g; _prev_d = _dates[_i]
# robust per-channel normalization (store stats so SkyGPT eval matches)
MOTION_MU = mot_all.mean(0); MOTION_SD = mot_all.std(0) + 1e-6
mot_all = ((mot_all - MOTION_MU) / MOTION_SD).astype(np.float32)
np.save(CHECKPOINT_DIR / "motion_norm.npy", np.stack([MOTION_MU, MOTION_SD]))
print(f"  motion features {mot_all.shape}  (saved motion_norm.npy)")
'''


SKIPPD_LATENTS_WRITE_CODE = '''\
# ==== SKIPP'D: CTI + covariates + write the splits/extended/latents contract ====
print("[SKIPPD-WRITE] CTI from latent velocity (per-day windowed) ...")
def _cti_per_day(z, days, w=10):
    out = np.zeros(len(z), np.float32)
    for d in np.unique(days):
        idx = np.where(days == d)[0]; zz = z[idx]
        for j in range(len(idx)):
            seg = zz[max(0, j-w):j+1]
            if len(seg) >= 3:
                out[idx[j]] = np.linalg.norm(np.var(np.diff(seg, axis=0), axis=0))
    return out
cti_all = _cti_per_day(Z_all, img_df["date"].values)
print(f"  CTI range [{cti_all.min():.2e}, {cti_all.max():.2e}]")

# Covariates: diurnal + seasonal harmonics + clear-sky ceiling + MOTION (4 dims).
_mod = img_df["mod"].values.astype(np.float32)
cov_all = np.stack([
    np.sin(2*np.pi*_mod/1440), np.cos(2*np.pi*_mod/1440),
    np.sin(2*np.pi*img_df["month"].values/12), np.cos(2*np.pi*img_df["month"].values/12),
    (img_df["cs"].values / float(img_df["cs"].max())),
], axis=1).astype(np.float32)
# append the optical-flow motion descriptor [dx, dy, mag, sun-region mag]
cov_all = np.concatenate([cov_all, mot_all], axis=1).astype(np.float32)
print(f"  covariates {cov_all.shape} (5 time/sky + {mot_all.shape[1]} motion)")

print("[SKIPPD-WRITE] writing splits + latents ...")
for s in ["train", "val", "test"]:
    m = (img_df["split"] == s).values
    sub = img_df[m].reset_index(drop=True)
    pd.DataFrame({
        "timestamp": sub["time"].dt.tz_localize(None),
        "ghi": sub["pv"].values, "ghi_clearsky": sub["cs"].values,
        "clear_sky_index": sub["kt"].values,
        "is_ramp": sub["is_ramp"].values,   # BASELINES/analysis read this from the parquet
    }).to_parquet(SPLITS_DIR / f"{s}.parquet")
    np.save(LATENT_DIR / f"{s}_latents.npy", Z_all[m])
    np.save(LATENT_DIR / f"{s}_cti.npy", cti_all[m])
    np.save(LATENT_DIR / f"{s}_ghi.npy", sub["pv"].values.astype(np.float32))
    np.save(LATENT_DIR / f"{s}_kt.npy", sub["kt"].values.astype(np.float32))
    np.save(LATENT_DIR / f"{s}_ghi_clearsky.npy", sub["cs"].values.astype(np.float32))
    np.save(LATENT_DIR / f"{s}_is_ramp.npy", sub["is_ramp"].values)
    np.save(LATENT_DIR / f"{s}_covariates.npy", cov_all[m])
    np.save(LATENT_DIR / f"{s}_physics_features.npy", cov_all[m])
    np.save(LATENT_DIR / f"{s}_image_features.npy", np.zeros((int(m.sum()), 10), np.float32))

# Extended = full PV labels (1-min) for sigma_pers + LSTM/CSDI baselines.
print("[SKIPPD-WRITE] writing extended (full PV labels) ...")
_labf = pd.concat([pd.read_parquet(SKIPPD_DIR / "labels" / "train-00000-of-00001.parquet"),
                   pd.read_parquet(SKIPPD_DIR / "labels" / "test-00000-of-00001.parquet")],
                  ignore_index=True)
_labf["time"] = pd.to_datetime(_labf["time"], utc=True).dt.tz_convert("US/Pacific")
_labf = _labf.sort_values("time").drop_duplicates("time").reset_index(drop=True)
_labf["pv"] = _labf["pv"].astype(np.float32)
_labf = skippd_clearsky(_labf)
_labf["date"] = _labf["time"].dt.date
_ld = np.array(sorted(_labf["date"].unique())); _n = len(_ld); _i1 = int(_n*0.7); _i2 = int(_n*0.85)
_labf["split"] = _labf["date"].map({**{d: "train" for d in _ld[:_i1]},
                                     **{d: "val" for d in _ld[_i1:_i2]},
                                     **{d: "test" for d in _ld[_i2:]}})
for s in ["train", "val", "test"]:
    sub = _labf[_labf["split"] == s]
    pd.DataFrame({"timestamp": sub["time"].dt.tz_localize(None),
                  "clear_sky_index": sub["kt"].values, "ghi": sub["pv"].values,
                  "ghi_clearsky": sub["cs"].values}).to_parquet(EXTENDED_DIR / f"{s}.parquet")
    print(f"  extended {s}: {len(sub):,} rows, {sub['date'].nunique()} days")
print("[SKIPPD-WRITE] contract written. Free image RAM.")
try:
    del Z_all, cti_all, cov_all, img_df, ds_all
    gc.collect()
except Exception:
    pass
'''


SKIPPD_HORIZON_OVERRIDE_CODE = '''\
# ==== SKIPP'D cadence override (1-min) — run AFTER LOAD_DATA_TOLERANT_CODE ====
# SKIPP'D is 1-minute cadence, so horizons are 1-min steps and PRIMARY_DT=60s.
# (CloudCV was 10s steps / PRIMARY_DT=10.) The architecture's h_norm=h/180 is a
# fixed normalization constant and stays consistent across the dataset + eval.
# Horizons include 15 min for the head-to-head with SkyGPT (their single horizon),
# plus 1/5/10 (our shorter-nowcast advantage) and 20/30 (breadth on the broader test).
HORIZONS = [1, 5, 10, 15, 20, 30]
HORIZON_MIN = {1: 1, 5: 5, 10: 10, 15: 15, 20: 20, 30: 30}
PRIMARY_DT = 60.0
# SEQ_LEN=16 -> 15-min history at 1-min cadence, matching SkyGPT's 16 log frames
# so the model can ingest their exact test windows without padding.
SEQ_LEN = 16
N_SAMPLES = 50
N_EVAL = min(2000, len(data["test"]["Z"]) - max(HORIZONS) - 1)
print(f"[SKIPPD] horizons={HORIZONS} min (1-min cadence), SEQ_LEN={SEQ_LEN}, PRIMARY_DT={PRIMARY_DT:.0f}s, N_EVAL={N_EVAL}")
print(f"[SKIPPD] target = rooftop PV (kW); kt = PV / clear-sky-PV envelope")
'''
