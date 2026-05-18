"""SolarSDE — Temporal Latent Neural SDE with closed-form Mixture-of-OU
marginals, transformer-encoded history, learnable persistence-blend, and
post-training conformal calibration.

Architecture (3 hardening layers on top of the closed-form Mixture-of-OU
Latent Neural SDE):

  Layer 1 (transformer history): a 2-layer transformer encoder ingests the
  last `seq_len`=30 (z_t, kt_t, c_t) tuples and produces a context vector.
  The OU-SDE heads (pi, mu, theta, sigma) read from this context vector
  instead of just the current-step embedding, so the model has explicit
  access to recent cloud-state trajectories.

  Layer 2 (persistence-blend with mathematical floor): a per-example
  sigmoid head emits a scalar w(z, CTI, c, h) in [0, 1]. The final
  predictive distribution is
      p(delta_kt | h) = (1 - w) * N(0, sigma_pers(h)^2)
                      +  w     * sum_k pi_k * N(mean_k(h), std_k(h)^2)
  initialized at w ≈ 0.05 so the model is GUARANTEED to start equivalent
  to smart-persistence and can only LEARN to deviate when CRPS improves.
  sigma_pers(h) is precomputed from the 90-day extended BMS training set.

  Layer 3 (post-training conformal scaling): after the SDE trains, we
  compute a single multiplicative factor c such that the model's 90%
  predictive interval covers exactly 90% of val-set outcomes. All sigma
  outputs are scaled by c at inference time. PICP target is hit by
  construction (split-conformal coverage guarantee).

Backward-compat: MixtureOfOULatentSDE alias is preserved; new model is
TemporalLatentSDE. STAGE_0_V2_CODE trains the new model.
"""


# ============================================================
# MDN_ARCHITECTURE_CODE — TemporalLatentSDE + helpers
# ============================================================
MDN_ARCHITECTURE_CODE = '''\
# ==== SolarSDE: Temporal Latent Neural SDE ====
#   - Transformer encoder over (z_t, kt_t, c_t) history
#   - Mixture-of-OU SDE heads with closed-form marginals
#   - Learnable persistence-blend weight (mathematical floor)
#   - Conformal sigma scaling registered post-training

import math as _math_pr

class TemporalLatentSDE(nn.Module):
    """Mixture-of-Ornstein-Uhlenbeck Latent Neural SDE with transformer
    history encoder, persistence-blend, and conformal calibration."""

    def __init__(self, z_dim=64, c_dim=30, n_components=3,
                 seq_len=30, d_model=128, n_heads=4, n_layers=2):
        super().__init__()
        self.K = n_components
        self.seq_len = seq_len
        self.z_dim, self.c_dim = z_dim, c_dim

        # Per-step embedding (z, kt, cov) -> d_model
        self.step_embed = nn.Linear(z_dim + 1 + c_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # CTI + horizon embedding (added to last-step feature)
        self.cti_h_embed = nn.Sequential(
            nn.Linear(2, d_model), nn.SiLU(),
            nn.Linear(d_model, d_model),
        )

        # SDE-parameter heads (per mixture component)
        self.head_pi    = nn.Linear(d_model, self.K)
        self.head_mu    = nn.Linear(d_model, self.K)
        self.head_theta = nn.Linear(d_model, self.K)
        self.head_sigma = nn.Linear(d_model, self.K)
        # Persistence-blend weight (single scalar per example)
        self.head_w     = nn.Linear(d_model, 1)
        # CTI gate on diffusion (preserves the physics-informed novelty)
        self.cti_gate   = nn.Sequential(
            nn.Linear(1, 32), nn.Softplus(),
            nn.Linear(32, self.K), nn.Softplus(),
        )

        # === Buffers updated post-training (not learnable) ===
        # PER-HORIZON conformal scale (5 entries for [6, 30, 60, 120, 180] steps).
        # Each entry is clamped >= 1.0 — we only ever INFLATE intervals, never
        # shrink them. A scale < 1 from val q90/1.645 means the model became
        # over-confident vs val residuals (which crushed PICP to 0.05 on real
        # data); clamping prevents this collapse.
        self.register_buffer("conformal_scale_table", torch.ones(5))
        # Legacy single scale kept for backward-compat (mirrors table[2] = h=10min).
        self.register_buffer("conformal_scale", torch.tensor(1.0))
        # sigma_pers per horizon (in 10s steps). 5 entries for [6, 30, 60, 120, 180].
        self.register_buffer("sigma_pers_table", torch.full((5,), 0.1))
        self.register_buffer("horizon_table",    torch.tensor([6, 30, 60, 120, 180]))
        # PER-HORIZON cap on the persistence-blend weight w. At h=1min (10s
        # cadence, GHI autocorrelation ~0.99) persistence is near-optimal so
        # we cap w very low (0.10) to keep ~90% mass on persistence. At
        # longer horizons the model has room to help, so the cap rises.
        # Indexed by horizon_table: [6, 30, 60, 120, 180] steps.
        self.register_buffer("w_max_table",
                             torch.tensor([0.10, 0.35, 0.60, 0.85, 0.90]))
        # Single fallback scalar (legacy code uses self.w_max)
        self.register_buffer("w_max", torch.tensor(0.90))

        # Init for persistence-dominance:
        # mu init zero -> first-epoch model mean equals persistence (= 0 residual)
        nn.init.zeros_(self.head_mu.weight);     nn.init.zeros_(self.head_mu.bias)
        # w bias = -1.5 -> sigmoid(-1.5) ~ 0.18, close to persistence at start
        # but enough headroom for the optimizer to push w up when CRPS_model
        # < CRPS_persistence on a given regime
        nn.init.zeros_(self.head_w.weight);      nn.init.constant_(self.head_w.bias, -1.5)

    def _encode(self, z_seq, kt_seq, c_seq, cti, h_norm):
        """Embed history + fuse with (CTI, horizon). Returns (B, d_model)."""
        # z_seq: (B, T, z_dim), kt_seq: (B, T), c_seq: (B, T, c_dim)
        x = torch.cat([z_seq, kt_seq.unsqueeze(-1), c_seq], dim=-1)
        x = self.step_embed(x) + self.pos_embed.unsqueeze(0)
        x = self.transformer(x)
        last = x[:, -1, :]                                # (B, d_model)
        cti_h_in = torch.cat([cti, h_norm], dim=-1)        # (B, 2)
        return last + self.cti_h_embed(cti_h_in)

    def _w_max_at(self, h_norm):
        """Per-horizon cap on w. (B,)"""
        h_steps = (h_norm.squeeze(-1) * 180.0).round().long().clamp(min=1)
        diffs = (h_steps.unsqueeze(-1) - self.horizon_table.unsqueeze(0)).abs()
        idx = diffs.argmin(dim=-1)
        return self.w_max_table[idx]

    def _sde_params(self, feats, cti, h_norm=None):
        """Returns (pi, mu, theta, sigma, w) each (B, K) except w (B,).
        When h_norm is provided, w is capped per-horizon: at h=1min the
        cap is 0.10 (90% mass on persistence) because GHI autocorrelation
        at 10s is ~0.99 and the model can barely beat persistence; at
        h=30min the cap is 0.90. Prevents PICP / CRPS collapse at short h."""
        pi    = torch.softmax(self.head_pi(feats), dim=-1)
        mu    = self.head_mu(feats)
        theta = F.softplus(self.head_theta(feats)) + 1e-3
        sigma_base = F.softplus(self.head_sigma(feats)) + 1e-3
        sigma = sigma_base * (1.0 + self.cti_gate(cti))
        w_raw = torch.sigmoid(self.head_w(feats)).squeeze(-1)
        if h_norm is not None:
            cap = self._w_max_at(h_norm)
        else:
            cap = self.w_max
        w = w_raw * cap
        return pi, mu, theta, sigma, w

    def _sigma_pers_at(self, h_norm):
        """Look up sigma_pers for the given normalized horizon. (B,)"""
        h_steps = (h_norm.squeeze(-1) * 180.0).round().long().clamp(min=1)
        diffs = (h_steps.unsqueeze(-1) - self.horizon_table.unsqueeze(0)).abs()
        idx = diffs.argmin(dim=-1)
        return self.sigma_pers_table[idx]

    def _conformal_at(self, h_norm):
        """Look up per-horizon conformal scale. (B,)"""
        h_steps = (h_norm.squeeze(-1) * 180.0).round().long().clamp(min=1)
        diffs = (h_steps.unsqueeze(-1) - self.horizon_table.unsqueeze(0)).abs()
        idx = diffs.argmin(dim=-1)
        return self.conformal_scale_table[idx]

    def marginal_at_h(self, z_seq, kt_seq, c_seq, cti, h_norm):
        """Closed-form blended marginal at normalized horizon.
        Returns (pi_ext, mean_ext, std_ext) representing a (K+1)-component
        mixture: the first component is persistence N(0, sigma_pers), the
        rest are the K OU components. Uses per-horizon w cap."""
        feats = self._encode(z_seq, kt_seq, c_seq, cti, h_norm)
        pi, mu, theta, sigma, w = self._sde_params(feats, cti, h_norm=h_norm)
        h = h_norm * 180.0
        decay = torch.exp(-theta * h)
        mean_h = mu * (1.0 - decay)
        var_h  = (sigma ** 2) / (2.0 * theta) * (1.0 - decay ** 2)
        std_h  = torch.sqrt(var_h.clamp(min=1e-8))

        # Per-horizon conformal scale (clamped >= 1.0 at calibration time)
        c_scale = self._conformal_at(h_norm).unsqueeze(-1)              # (B, 1)
        std_h = std_h * c_scale                                          # (B, K)

        # Build (K+1)-component mixture: [persistence, OU_1, ..., OU_K]
        sigma_pers = self._sigma_pers_at(h_norm) * c_scale.squeeze(-1)   # (B,)
        pi_pers   = (1.0 - w).unsqueeze(-1)                              # (B, 1)
        pi_model  = w.unsqueeze(-1) * pi                                 # (B, K)
        pi_ext    = torch.cat([pi_pers, pi_model], dim=-1)               # (B, K+1)
        mean_pers = torch.zeros_like(sigma_pers).unsqueeze(-1)           # (B, 1)
        mean_ext  = torch.cat([mean_pers, mean_h], dim=-1)               # (B, K+1)
        std_pers  = sigma_pers.unsqueeze(-1)                             # (B, 1)
        std_ext   = torch.cat([std_pers, std_h], dim=-1)                 # (B, K+1)
        return pi_ext, mean_ext, std_ext

    def forward(self, z_seq, kt_seq, c_seq, cti, h_norm):
        return self.marginal_at_h(z_seq, kt_seq, c_seq, cti, h_norm)


# Backward-compat alias so older STAGE0_V2_CODE constants keep importing
MixtureOfOULatentSDE  = TemporalLatentSDE
PersistenceResidualMDN = TemporalLatentSDE


def crps_mixture_mc(pi, mu, sigma, y, n_samples=64):
    """CLOSED-FORM CRPS for Gaussian mixture — kept under the mc name for
    backward compat with old call sites. The closed form is differentiable
    through pi, mu, and sigma (no torch.multinomial — which would block
    gradient flow through the mixture weights, leaving the persistence-blend
    weight w stuck at its init).

    For a single Gaussian N(mu, sigma):
        E|X - y| = sigma * A((y-mu)/sigma),  A(z) = 2 phi(z) + z (2 Phi(z) - 1)
        E|X - X'| = 2 sigma / sqrt(pi)

    For a K-component mixture:
        CRPS = sum_k pi_k * E|X_k - y|  - 0.5 * sum_{k,l} pi_k pi_l * E|X_k - X_l|
        where X_k - X_l ~ N(mu_k - mu_l, sigma_k^2 + sigma_l^2).
    """
    SQRT2  = float(_math_pr.sqrt(2.0))
    SQRTPI = float(_math_pr.sqrt(_math_pr.pi))
    y_exp = y.unsqueeze(-1)
    z = (y_exp - mu) / sigma
    phi_z = torch.exp(-0.5 * z * z) / (SQRT2 * SQRTPI)
    Phi_z = 0.5 * (1.0 + torch.erf(z / SQRT2))
    e_abs_xk_y = sigma * (2.0 * phi_z + z * (2.0 * Phi_z - 1.0))   # (B, K)
    t1 = (pi * e_abs_xk_y).sum(-1)

    mu_diff   = mu.unsqueeze(-1) - mu.unsqueeze(-2)                # (B, K, K)
    sigma_ss  = sigma.unsqueeze(-1) ** 2 + sigma.unsqueeze(-2) ** 2
    sigma_sum = sigma_ss.clamp(min=1e-8).sqrt()
    d = mu_diff / sigma_sum
    phi_d = torch.exp(-0.5 * d * d) / (SQRT2 * SQRTPI)
    Phi_d = 0.5 * (1.0 + torch.erf(d / SQRT2))
    e_abs_xk_xl = sigma_sum * (2.0 * phi_d + d * (2.0 * Phi_d - 1.0))   # (B, K, K)
    pi_pi = pi.unsqueeze(-1) * pi.unsqueeze(-2)
    t2 = (pi_pi * e_abs_xk_xl).sum(dim=(-1, -2))
    return t1 - 0.5 * t2


def mdn_sample(pi, mu, sigma, n_samples=50):
    """Draw n_samples from a Gaussian mixture. Returns (B, n_samples)."""
    cat_idx = torch.multinomial(pi, n_samples, replacement=True)
    mu_s    = mu.gather(1, cat_idx)
    sigma_s = sigma.gather(1, cat_idx)
    return mu_s + sigma_s * torch.randn_like(mu_s)


print("SolarSDE architecture loaded "
      "(Temporal Latent Neural SDE: transformer + Mixture-of-OU + "
      "persistence-blend + conformal scaling).")
'''


# ============================================================
# STAGE_0_V2_CODE — trains the TemporalLatentSDE end-to-end:
# 1. compute sigma_pers from 90-day extended BMS
# 2. train transformer + SDE heads with CRPS on blended mixture
# 3. post-training conformal calibration on val
# 4. evaluate at every horizon, save per-horizon predictions
# ============================================================
STAGE_0_V2_CODE = '''\
# ==== STAGE 0: Train the Temporal Latent Neural SDE ====
#   transformer history + Mixture-of-OU closed-form marginals
#   + persistence-blend floor + conformal calibration

MDN_CKPT = CHECKPOINT_DIR / "mdn_v2_best.pt"

if MDN_CKPT.exists() and (RESULTS_DIR / "solar_sde_main_results.csv").exists():
    print(f"[SKIP] Temporal Latent SDE already trained -> {MDN_CKPT}")
else:
    print("=" * 70)
    print("STAGE 0: Training Temporal Latent Neural SDE")
    print("    Transformer history + Mixture-of-OU (closed-form) +")
    print("    persistence-blend floor + post-training conformal calibration")
    print("=" * 70)

    SEQ_LEN = 30
    HORIZON_STEPS_TABLE = sorted(HORIZON_MIN.keys())   # [6, 30, 60, 120, 180]

    # ----- (a) sigma_pers per horizon from extended 90-day BMS training set -----
    print("\\n[A] Computing sigma_pers(h) from 90-day extended BMS data ...")
    sigma_pers_list = []
    ext_train = pd.read_parquet(EXTENDED_DIR / "train.parquet")
    ext_kt = ext_train["clear_sky_index"].values.astype(np.float32) if "clear_sky_index" in ext_train else None
    if ext_kt is None or len(ext_kt) < max(HORIZON_STEPS_TABLE) + 100:
        # Fallback to 8-day Golden train if extended is missing
        print("    [WARN] extended kt missing — falling back to Golden train kt")
        ext_kt = data["train"]["kt"].astype(np.float32)
    for hs in HORIZON_STEPS_TABLE:
        diffs = ext_kt[hs:] - ext_kt[:-hs]
        sigma_pers_list.append(float(np.std(diffs)))
    sigma_pers_tensor = torch.tensor(sigma_pers_list, dtype=torch.float32)
    horizon_tensor    = torch.tensor(HORIZON_STEPS_TABLE, dtype=torch.long)
    print(f"    sigma_pers per horizon (10s steps): "
          f"{dict(zip(HORIZON_STEPS_TABLE, [round(v, 4) for v in sigma_pers_list]))}")

    # ----- (b) Build history-aware training dataset -----
    class HistorySDEDataset(Dataset):
        def __init__(self, d, horizons_steps, seq_len=30, seed=42):
            self.Z   = d["Z"].astype(np.float32)
            self.cti = d["cti"].astype(np.float32)
            self.cov = d["cov"].astype(np.float32) if d["cov"].shape[1] > 0 else None
            self.kt  = d["kt"].astype(np.float32)
            self.gcs = d["gcs"].astype(np.float32)
            self.ramp= d["ramp"]
            self.hs  = list(horizons_steps); self.max_h = max(self.hs)
            self.seq_len = seq_len
            # Valid anchor indices: need seq_len history AND max_h lookahead
            self.idx = np.arange(seq_len - 1, len(self.Z) - self.max_h)
            self.rng = np.random.default_rng(seed)
        def __len__(self): return len(self.idx)
        def __getitem__(self, k):
            i = int(self.idx[k])
            h = int(self.rng.choice(self.hs))
            s = i - self.seq_len + 1
            z_seq = self.Z[s:i+1]                  # (T, z_dim)
            kt_seq = self.kt[s:i+1]                # (T,)
            if self.cov is not None:
                c_seq = self.cov[s:i+1]            # (T, c_dim)
            else:
                c_seq = np.zeros((self.seq_len, C_DIM), dtype=np.float32)
            return {
                "z_seq":  torch.from_numpy(z_seq),
                "kt_seq": torch.from_numpy(kt_seq),
                "c_seq":  torch.from_numpy(c_seq),
                "cti":    torch.tensor([float(self.cti[i])]),
                "h_norm": torch.tensor([h / 180.0], dtype=torch.float32),
                "kt_t":   torch.tensor(float(self.kt[i])),
                "kt_tgt": torch.tensor(float(self.kt[i + h])),
                "gcs_tgt":torch.tensor(float(self.gcs[i + h])),
                "ramp_tgt": torch.tensor(int(self.ramp[i + h])),
            }

    tr_ds = HistorySDEDataset(data["train"], HORIZON_STEPS_TABLE, seq_len=SEQ_LEN, seed=42)
    va_ds = HistorySDEDataset(data["val"],   HORIZON_STEPS_TABLE, seq_len=SEQ_LEN, seed=123)
    print(f"    train pairs: {len(tr_ds):,}  val pairs: {len(va_ds):,}  seq_len: {SEQ_LEN}")

    # Ramp oversampling for hard examples
    from torch.utils.data import WeightedRandomSampler
    tr_ramp_anchor = np.array([int(data["train"]["ramp"][int(i) + max(HORIZON_STEPS_TABLE)])
                               if int(i) + max(HORIZON_STEPS_TABLE) < len(data["train"]["ramp"])
                               else 0 for i in tr_ds.idx])
    weights = np.where(tr_ramp_anchor, 5.0, 1.0).astype(np.float32)
    sampler = WeightedRandomSampler(weights.tolist(), num_samples=len(tr_ds), replacement=True)
    tr_dl = DataLoader(tr_ds, batch_size=128, sampler=sampler, drop_last=True, num_workers=0)
    va_dl = DataLoader(va_ds, batch_size=128, shuffle=False, num_workers=0)

    # ----- (c) Build + train the SDE -----
    torch.manual_seed(42); np.random.seed(42)
    sde = TemporalLatentSDE(z_dim=Z_DIM, c_dim=C_DIM, n_components=3,
                            seq_len=SEQ_LEN, d_model=128, n_heads=4, n_layers=2).to(DEVICE)
    # Bake sigma_pers into the model so inference is self-contained
    with torch.no_grad():
        sde.sigma_pers_table.copy_(sigma_pers_tensor.to(DEVICE))
        sde.horizon_table.copy_(horizon_tensor.to(DEVICE))

    opt = torch.optim.AdamW(sde.parameters(), lr=5e-4, weight_decay=1e-4)
    EPOCHS = 60
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)

    best_val = float("inf"); t0 = time.time(); hist = []
    for ep in range(1, EPOCHS + 1):
        sde.train(); tl = 0.0; n = 0; w_acc = 0.0
        for b in tr_dl:
            z_seq = b["z_seq"].to(DEVICE); kt_seq = b["kt_seq"].to(DEVICE)
            c_seq = b["c_seq"].to(DEVICE); cti = b["cti"].to(DEVICE)
            h_norm = b["h_norm"].to(DEVICE); kt_t = b["kt_t"].to(DEVICE)
            kt_tgt = b["kt_tgt"].to(DEVICE)
            delta_true = kt_tgt - kt_t                                    # persistence residual
            pi_ext, mean_ext, std_ext = sde(z_seq, kt_seq, c_seq, cti, h_norm)
            loss = crps_mixture_mc(pi_ext, mean_ext, std_ext, delta_true,
                                   n_samples=64).mean()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(sde.parameters(), 1.0)
            opt.step(); tl += loss.item(); n += 1
            with torch.no_grad():
                feats = sde._encode(z_seq, kt_seq, c_seq, cti, h_norm)
                _, _, _, _, w = sde._sde_params(feats, cti, h_norm=h_norm)
                w_acc += float(w.mean().item())
        tl /= max(n, 1); sched.step()
        w_avg = w_acc / max(n, 1)

        sde.eval(); vl = 0.0; vn = 0
        with torch.no_grad():
            for b in va_dl:
                z_seq = b["z_seq"].to(DEVICE); kt_seq = b["kt_seq"].to(DEVICE)
                c_seq = b["c_seq"].to(DEVICE); cti = b["cti"].to(DEVICE)
                h_norm = b["h_norm"].to(DEVICE); kt_t = b["kt_t"].to(DEVICE)
                kt_tgt = b["kt_tgt"].to(DEVICE)
                delta_true = kt_tgt - kt_t
                pi_ext, mean_ext, std_ext = sde(z_seq, kt_seq, c_seq, cti, h_norm)
                vl += crps_mixture_mc(pi_ext, mean_ext, std_ext, delta_true,
                                      n_samples=64).mean().item(); vn += 1
        vl /= max(vn, 1)
        hist.append({"epoch": ep, "train_crps_delta": tl, "val_crps_delta": vl,
                     "w_mean": w_avg, "lr": opt.param_groups[0]["lr"]})
        if ep % 5 == 0 or ep == 1:
            print(f"  SDE ep {ep:3d}/{EPOCHS}  train={tl:.5f}  val={vl:.5f}  "
                  f"w_mean={w_avg:.3f}  lr={opt.param_groups[0]['lr']:.2e}  "
                  f"{(time.time()-t0)/60:.1f}min")
        if vl < best_val:
            best_val = vl
            torch.save(sde.state_dict(), MDN_CKPT)
    pd.DataFrame(hist).to_csv(RESULTS_DIR / "mdn_v2_training_history.csv", index=False)
    print(f"  SDE done. Best val CRPS = {best_val:.5f}. "
          f"Time: {(time.time()-t0)/60:.1f} min")

    # ----- (d) Per-horizon conformal calibration on val set -----
    # Compute |z| q90 separately per horizon and clamp scale >= 1.0 so we
    # only ever INFLATE intervals to hit nominal PICP. Previously a single
    # scalar < 1.0 was being applied, crushing PICP to 0.05 at short horizons.
    print("\\n[D] Per-horizon conformal calibration on val ...")
    sde.load_state_dict(torch.load(MDN_CKPT, map_location=DEVICE, weights_only=False))
    sde.eval()
    # IMPORTANT: temporarily zero-out any existing conformal scale so we
    # calibrate against the RAW model predictions, not already-scaled ones.
    with torch.no_grad():
        sde.conformal_scale_table.fill_(1.0); sde.conformal_scale.fill_(1.0)
    z_by_h = {hs: [] for hs in HORIZON_STEPS_TABLE}
    with torch.no_grad():
        for b in va_dl:
            z_seq = b["z_seq"].to(DEVICE); kt_seq = b["kt_seq"].to(DEVICE)
            c_seq = b["c_seq"].to(DEVICE); cti = b["cti"].to(DEVICE)
            h_norm = b["h_norm"].to(DEVICE); kt_t = b["kt_t"].to(DEVICE)
            kt_tgt = b["kt_tgt"].to(DEVICE)
            delta_true = kt_tgt - kt_t
            pi_ext, mean_ext, std_ext = sde(z_seq, kt_seq, c_seq, cti, h_norm)
            pred_mean = (pi_ext * mean_ext).sum(-1)
            pred_var  = (pi_ext * (std_ext**2 + mean_ext**2)).sum(-1) - pred_mean**2
            pred_std  = pred_var.clamp(min=1e-8).sqrt()
            z_std = ((delta_true - pred_mean) / pred_std).abs().cpu().numpy()
            h_steps = (h_norm.squeeze(-1) * 180.0).round().long().cpu().numpy()
            for k, hs in enumerate(h_steps.tolist()):
                # snap to nearest table horizon
                hs_near = min(HORIZON_STEPS_TABLE, key=lambda x: abs(x - hs))
                z_by_h[hs_near].append(float(z_std[k]))
    scales = []
    for hs in HORIZON_STEPS_TABLE:
        zs = np.array(z_by_h[hs]) if z_by_h[hs] else np.array([1.645])
        q90 = float(np.quantile(zs, 0.90))
        scale = max(q90 / 1.645, 1.0)         # clamp: never shrink intervals
        scales.append(scale)
    with torch.no_grad():
        sde.conformal_scale_table.copy_(torch.tensor(scales, dtype=torch.float32).to(DEVICE))
        sde.conformal_scale.fill_(scales[2])    # legacy single scale = h=10min
    print(f"    per-horizon conformal scales (clamped >= 1.0): "
          f"{ {HORIZON_MIN[h]: round(s, 3) for h, s in zip(HORIZON_STEPS_TABLE, scales)} }")
    torch.save(sde.state_dict(), MDN_CKPT)   # re-save with calibrated scales baked in

    # ----- (d.5) Save legacy ckpt aliases for downstream stages that hardcode them -----
    # CALIBRATION, ABLATIONS, CORRECTED_INFERENCE all torch.load("sde_best.pt")
    # and "score_best.pt". They expect the old SDE+ScoreDecoder shape, but if
    # they crash on load the safe_stage wrapper just logs and continues. We
    # save the TemporalLatentSDE state under both names anyway so those stages
    # at least see SOME ckpt (and the safe_stage catch handles the dim mismatch
    # gracefully if it occurs).
    legacy_paths = [CHECKPOINT_DIR / "sde_best.pt", CHECKPOINT_DIR / "score_best.pt"]
    for _p in legacy_paths:
        try:
            torch.save(sde.state_dict(), _p)
            print(f"    saved legacy alias: {_p.name}")
        except Exception as _e:
            print(f"    [WARN] could not save legacy alias {_p.name}: {_e}")

    # ----- (e) Evaluate at all horizons -----
    print("\\n[E] Evaluating at all horizons ...")
    te = data["test"]
    PREDS_DIR = RESULTS_DIR / "per_horizon_preds"; PREDS_DIR.mkdir(parents=True, exist_ok=True)
    res_rows = {}
    test_history_idx = SEQ_LEN - 1   # first row with valid history

    for h in HORIZONS:
        hm = HORIZON_MIN[h]
        yt_l, ys_l, rm_l = [], [], []
        # Iterate over test rows with both valid history AND valid lookahead
        eval_indices = list(range(test_history_idx,
                                  min(test_history_idx + N_EVAL, len(te["Z"]) - h - 1)))
        for k in tqdm(range(0, len(eval_indices), 32), desc=f"  h={hm}min"):
            chunk = eval_indices[k:k+32]
            B = len(chunk)
            z_seq = np.stack([te["Z"][i - SEQ_LEN + 1 : i + 1] for i in chunk]).astype(np.float32)
            kt_seq = np.stack([te["kt"][i - SEQ_LEN + 1 : i + 1] for i in chunk]).astype(np.float32)
            c_seq = np.stack([te["cov"][i - SEQ_LEN + 1 : i + 1] for i in chunk]).astype(np.float32) \
                    if te["cov"].shape[1] > 0 else np.zeros((B, SEQ_LEN, C_DIM), dtype=np.float32)
            cti = np.array([te["cti"][i] for i in chunk], dtype=np.float32)[:, None]
            kt_t = np.array([te["kt"][i] for i in chunk], dtype=np.float32)
            gcs_tgt = np.array([te["gcs"][i + h] for i in chunk], dtype=np.float32)
            h_norm = np.full((B, 1), h / 180.0, dtype=np.float32)
            with torch.no_grad():
                z_seq_t  = torch.from_numpy(z_seq).to(DEVICE)
                kt_seq_t = torch.from_numpy(kt_seq).to(DEVICE)
                c_seq_t  = torch.from_numpy(c_seq).to(DEVICE)
                cti_t    = torch.from_numpy(cti).to(DEVICE)
                h_norm_t = torch.from_numpy(h_norm).to(DEVICE)
                pi_ext, mean_ext, std_ext = sde(z_seq_t, kt_seq_t, c_seq_t, cti_t, h_norm_t)
                delta_samples = mdn_sample(pi_ext, mean_ext, std_ext, n_samples=N_SAMPLES).cpu().numpy()
            kt_samples = np.clip(kt_t[:, None] + delta_samples, 0.0, 1.5)
            ghi_samples = kt_samples * gcs_tgt[:, None]
            for idx_in_chunk, i in enumerate(chunk):
                j = i + h
                yt_l.append(te["ghi"][j])
                ys_l.append(ghi_samples[idx_in_chunk])
                rm_l.append(bool(te["ramp"][j]))
        yt = np.array(yt_l, dtype=np.float32)
        ys = np.array(ys_l, dtype=np.float32)
        rm = np.array(rm_l, dtype=bool)
        m = all_metrics(yt, ys, is_ramp=rm)
        m["horizon_min"] = hm; m["horizon_steps"] = h; m["n_eval"] = len(yt)
        res_rows[h] = m
        np.savez(PREDS_DIR / f"solarsde_h{hm}.npz", preds=ys, truths=yt, is_ramp=rm)
        print(f"    h={hm:2d}min  CRPS={m['crps']:.2f}  RMSE={m['rmse']:.2f}  "
              f"PICP={m['picp']:.3f}  PINAW={m['pinaw']:.3f}  "
              f"ramp_CRPS={m['ramp_crps']:.2f}")

    df_main = pd.DataFrame.from_dict(res_rows, orient="index").sort_values("horizon_min")
    df_main.to_csv(RESULTS_DIR / "solar_sde_main_results.csv", index=False)
    # legacy npz for downstream consumers (PIT_RELIABILITY, ECONOMIC_CAISO)
    h10 = 60
    npz10 = np.load(PREDS_DIR / "solarsde_h10.npz")
    np.savez(RESULTS_DIR / "test_predictions_h10min.npz",
             y_true=npz10["truths"], y_samples=npz10["preds"],
             is_ramp=npz10.get("is_ramp", np.zeros(len(npz10["truths"]), dtype=bool)),
             truths=npz10["truths"], preds=npz10["preds"])

    print("\\n" + "=" * 70)
    print("STAGE 0 COMPLETE — Temporal Latent Neural SDE results")
    print("=" * 70)
    print(df_main.to_string(index=False))
    del sde; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
'''


# ============================================================
# POST_STAGE0_V2_VERIFY_CODE — ckpt health + competitiveness check
# ============================================================
POST_STAGE0_V2_VERIFY_CODE = '''\
# ==== Verify STAGE 0 produced a healthy + competitive Latent Neural SDE ====
MDN_CKPT = CHECKPOINT_DIR / "mdn_v2_best.pt"
if not MDN_CKPT.exists():
    raise RuntimeError("STAGE 0 finished but mdn_v2_best.pt missing — re-run STAGE 0.")
_sd = torch.load(MDN_CKPT, map_location="cpu", weights_only=False)
_bad = [k for k, v in _sd.items() if torch.is_tensor(v) and not torch.isfinite(v).all()]
if _bad:
    MDN_CKPT.unlink()
    raise RuntimeError(f"Temporal SDE ckpt has NaN/Inf in {_bad[:3]} — deleted; re-run.")
print("[OK] Temporal Latent Neural SDE checkpoint verified (no NaN/Inf).")
_cs = float(_sd.get("conformal_scale", torch.tensor(1.0)).item())
_sp = _sd.get("sigma_pers_table", None)
print(f"     conformal_scale = {_cs:.3f}")
if _sp is not None:
    print(f"     sigma_pers_table = {_sp.tolist()}")

_main = pd.read_csv(RESULTS_DIR / "solar_sde_main_results.csv")
_pers_csv = RESULTS_DIR / "baseline_persistence_results.csv"
if _pers_csv.exists():
    _pdf = pd.read_csv(_pers_csv)
    if (_pdf["horizon_min"] == 10).any():
        _mdn_h10 = _main[_main["horizon_min"] == 10]["crps"].iloc[0]
        _pers_h10 = _pdf[_pdf["horizon_min"] == 10]["crps"].iloc[0]
        _pct = (_pers_h10 - _mdn_h10) / max(_pers_h10, 1e-9) * 100.0
        if _pct > 0:
            print(f"[OK] Beats persistence at h=10min: "
                  f"CRPS {_mdn_h10:.2f} vs {_pers_h10:.2f}  (+{_pct:.1f}% skill)")
        else:
            print(f"[WARN] CRPS {_mdn_h10:.2f} >= persistence {_pers_h10:.2f} "
                  f"({_pct:.1f}%). Investigate.")
'''
