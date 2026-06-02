"""SolarSDE-Rollout — the latent-rollout variant (notebook 10).

Where the closed-form model (notebook 09) reads a closed-form OU marginal at
horizon h, this model *evolves the cloud-state latent forward* with a learned
neural SDE (Euler-Maruyama) and decodes the future latent to a PV distribution —
the original CLAUDE.md design. This explicitly models cloud-state EVOLUTION
(advection), the mechanism SkyGPT gets by generating future sky images, but in
latent space (no pixel generation).

Drop-in by design: `RolloutLatentSDE.forward(z_seq, kt_seq, c_seq, cti, h_norm)`
returns the SAME `(pi_ext, mean_ext, std_ext)` mixture-over-delta-kt as the
closed-form model, with the SAME calibration buffers, and is aliased to
`TemporalLatentSDE`. So STAGE_0_V2_CODE, the SkyGPT eval, sampling-efficiency,
compute-cost, and cross-validation all run UNCHANGED — only the marginal is now
produced by rolling the SDE forward. Ablations need a rollout-specific version
(ABLATIONS_ROLLOUT_CODE) since the internal components differ.
"""

ROLLOUT_ARCH_CODE = '''\
# ==== SolarSDE-Rollout: latent neural SDE rolled forward + decoder ====
import math as _math_pr

class RolloutLatentSDE(nn.Module):
    """Latent neural SDE: encode history -> roll z forward to t+h via learned
    drift + CTI-gated diffusion (Euler-Maruyama, n_paths sample paths) ->
    decode each future latent to a delta-kt distribution. Blended with a
    persistence anchor and post-hoc conformally calibrated, exactly like the
    closed-form model, so the (pi, mean, std) interface is identical."""

    def __init__(self, z_dim=64, c_dim=28, n_components=3,
                 seq_len=16, d_model=128, n_heads=4, n_layers=2,
                 n_horizons=6, n_paths=16):
        super().__init__()
        self.z_dim, self.c_dim = z_dim, c_dim
        self.seq_len = seq_len
        self.n_horizons = n_horizons
        self.n_paths = n_paths

        # History encoder (transformer over (z, kt, c) tuples)
        self.step_embed = nn.Linear(z_dim + 1 + c_dim, d_model)
        self.pos_embed  = nn.Parameter(torch.randn(seq_len, d_model) * 0.02)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model,
            dropout=0.1, batch_first=True, norm_first=True, activation="gelu")
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.cti_h_embed = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.ctx_proj = nn.Linear(d_model, d_model)
        self.d_model = d_model

        # Drift network: how the latent evolves per step. [z, ctx, step_frac] -> dz
        self.drift = nn.Sequential(
            nn.Linear(z_dim + d_model + 1, 256), nn.SiLU(),
            nn.Linear(256, 256), nn.SiLU(),
            nn.Linear(256, z_dim))
        # CTI-gated diffusion: state features * Softplus(CTI gate) -> per-dim sigma
        self.diff_state = nn.Linear(z_dim, 64)
        self.diff_cti   = nn.Sequential(nn.Linear(1, 64), nn.Softplus())
        self.diff_out   = nn.Sequential(nn.Linear(64, z_dim), nn.Softplus())
        # Decoder: future latent + ctx -> (mean, log_std) of delta-kt
        self.dec = nn.Sequential(
            nn.Linear(z_dim + d_model, 128), nn.SiLU(),
            nn.Linear(128, 64), nn.SiLU(),
            nn.Linear(64, 2))
        # Persistence-blend weight (per example)
        self.head_w = nn.Linear(d_model, 1)

        # ---- Calibration / blend buffers (identical layout to the closed-form model) ----
        H = n_horizons
        self.register_buffer("conformal_scale_table", torch.ones(H))
        self.register_buffer("conformal_scale", torch.tensor(1.0))
        self.register_buffer("conformal_cti_table", torch.ones(H, 4))
        self.register_buffer("conformal_cti_cuts", torch.zeros(3))
        self.register_buffer("sigma_pers_table", torch.full((H,), 0.1))
        self.register_buffer("horizon_table", torch.zeros(H, dtype=torch.long))
        self.register_buffer("w_max_table", torch.linspace(0.30, 0.95, H))
        self.register_buffer("w_max", torch.tensor(0.95))
        self.sigma_pers_cti_alpha = nn.Parameter(torch.tensor(0.5))

        nn.init.zeros_(self.head_w.weight); nn.init.constant_(self.head_w.bias, -1.0)
        # small initial diffusion so early rollout is near-deterministic
        nn.init.zeros_(self.diff_out[0].weight); nn.init.constant_(self.diff_out[0].bias, -3.0)
        self.STEP_DT = 1.0 / 30.0   # normalized Euler step (horizon measured in 1-min steps)

    def _encode(self, z_seq, kt_seq, c_seq, cti, h_norm):
        # Returns the context vector ONLY (single tensor) to match the closed-form
        # model's interface (STAGE_0_V2_CODE's w_mean diagnostic calls _encode then
        # _sde_params). marginal_at_h takes z0 = z_seq[:, -1, :] directly.
        x = torch.cat([z_seq, kt_seq.unsqueeze(-1), c_seq], dim=-1)
        x = self.step_embed(x) + self.pos_embed.unsqueeze(0)
        x = self.transformer(x)
        last = x[:, -1, :] + self.cti_h_embed(torch.cat([cti, h_norm], dim=-1))
        return self.ctx_proj(last)                     # (B, d_model)

    def _sde_params(self, feats, cti, h_norm=None):
        """Compatibility shim for STAGE_0_V2_CODE's w_mean diagnostic. Only the
        persistence-blend weight w is meaningful here; the OU params don't exist
        in the rollout model so we return None for them."""
        w_raw = torch.sigmoid(self.head_w(feats)).squeeze(-1)
        cap = self._w_max_at(h_norm) if h_norm is not None else self.w_max
        return None, None, None, None, w_raw * cap

    def _diffusion(self, z, cti):
        return self.diff_out(self.diff_state(z).mul(self.diff_cti(cti))) + 1e-4

    def _w_max_at(self, h_norm):
        hs = (h_norm.squeeze(-1) * 180.0).round().long().clamp(min=1)
        idx = (hs.unsqueeze(-1) - self.horizon_table.unsqueeze(0)).abs().argmin(-1)
        return self.w_max_table[idx]

    def _sigma_pers_at(self, h_norm):
        hs = (h_norm.squeeze(-1) * 180.0).round().long().clamp(min=1)
        idx = (hs.unsqueeze(-1) - self.horizon_table.unsqueeze(0)).abs().argmin(-1)
        return self.sigma_pers_table[idx]

    def _conformal_at(self, h_norm):
        hs = (h_norm.squeeze(-1) * 180.0).round().long().clamp(min=1)
        idx = (hs.unsqueeze(-1) - self.horizon_table.unsqueeze(0)).abs().argmin(-1)
        return self.conformal_scale_table[idx]

    def _cti_bucket_at(self, cti, h_norm):
        hs = (h_norm.squeeze(-1) * 180.0).round().long().clamp(min=1)
        h_idx = (hs.unsqueeze(-1) - self.horizon_table.unsqueeze(0)).abs().argmin(-1)
        c_idx = (cti.squeeze(-1).unsqueeze(-1) > self.conformal_cti_cuts.unsqueeze(0)).sum(-1).long().clamp(0, 3)
        return self.conformal_cti_table[h_idx, c_idx]

    def marginal_at_h(self, z_seq, kt_seq, c_seq, cti, h_norm):
        """Roll the latent forward to t+h (n_paths sample paths), decode each to
        a delta-kt component, blend with persistence, conformally scale.
        Returns (pi_ext, mean_ext, std_ext) over delta-kt — same interface."""
        ctx = self._encode(z_seq, kt_seq, c_seq, cti, h_norm)
        z0 = z_seq[:, -1, :]
        B = z0.shape[0]; P = self.n_paths; dev = z0.device
        # Per-ROW horizon (the training dataset mixes horizons within a batch):
        # roll everyone to Hmax but snapshot each path's latent at ITS OWN h_steps.
        h_steps = (h_norm.squeeze(-1) * 180.0).round().long().clamp(min=1)        # (B,)
        Hmax = int(h_steps.max().item())
        h_steps_P = h_steps.unsqueeze(1).expand(B, P).reshape(B * P)              # (B*P,)
        z   = z0.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
        ctxP = ctx.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
        ctiP = cti.unsqueeze(1).expand(B, P, -1).reshape(B * P, -1)
        z_final = z.clone()
        for step in range(Hmax):
            sf = torch.full((B * P, 1), step * self.STEP_DT, device=dev)
            drift = self.drift(torch.cat([z, ctxP, sf], dim=-1))
            sig   = self._diffusion(z, ctiP)
            z = z + drift * self.STEP_DT + sig * (self.STEP_DT ** 0.5) * torch.randn_like(z)
            done = ((step + 1) == h_steps_P).unsqueeze(-1)                        # rows finishing now
            z_final = torch.where(done, z, z_final)
        dec = self.dec(torch.cat([z_final, ctxP], dim=-1))
        delta_mean = dec[:, 0].reshape(B, P)
        delta_std  = (F.softplus(dec[:, 1]) + 1e-3).reshape(B, P)

        # conformal scaling (per-horizon x per-CTI-bucket), identical to closed-form
        c_scale = (self._conformal_at(h_norm) * self._cti_bucket_at(cti, h_norm)).unsqueeze(-1)
        std_paths = delta_std * c_scale                                  # (B, P)

        # persistence blend: [persistence N(0, sigma_pers)] + P rolled components
        w_raw = torch.sigmoid(self.head_w(ctx)).squeeze(-1)
        w = w_raw * self._w_max_at(h_norm)                               # (B,)
        sigma_pers = self._sigma_pers_at(h_norm) * (1.0 + cti.squeeze(-1) * F.softplus(self.sigma_pers_cti_alpha))
        sigma_pers = sigma_pers * c_scale.squeeze(-1)
        pi_pers  = (1.0 - w).unsqueeze(-1)                               # (B,1)
        pi_paths = (w.unsqueeze(-1) / P).expand(B, P)                    # (B,P)
        pi_ext   = torch.cat([pi_pers, pi_paths], dim=-1)
        mean_ext = torch.cat([torch.zeros(B, 1, device=dev), delta_mean], dim=-1)
        std_ext  = torch.cat([sigma_pers.unsqueeze(-1), std_paths], dim=-1)
        return pi_ext, mean_ext, std_ext

    def forward(self, z_seq, kt_seq, c_seq, cti, h_norm):
        return self.marginal_at_h(z_seq, kt_seq, c_seq, cti, h_norm)


# Alias so STAGE_0_V2_CODE / SkyGPT eval / CV / sampling / compute all reuse unchanged.
TemporalLatentSDE = RolloutLatentSDE
MixtureOfOULatentSDE = RolloutLatentSDE


def crps_mixture_mc(pi, mu, sigma, y, n_samples=64):
    """Closed-form Gaussian-mixture CRPS (differentiable through pi/mu/sigma).
    Same as the closed-form model — the rollout produces a (P+1)-component
    Gaussian mixture per example, so the identical CRPS applies."""
    SQRT2  = float(_math_pr.sqrt(2.0)); SQRTPI = float(_math_pr.sqrt(_math_pr.pi))
    y_exp = y.unsqueeze(-1)
    z = (y_exp - mu) / sigma
    phi_z = torch.exp(-0.5 * z * z) / (SQRT2 * SQRTPI)
    Phi_z = 0.5 * (1.0 + torch.erf(z / SQRT2))
    e_abs_xk_y = sigma * (2.0 * phi_z + z * (2.0 * Phi_z - 1.0))
    t1 = (pi * e_abs_xk_y).sum(-1)
    mu_diff   = mu.unsqueeze(-1) - mu.unsqueeze(-2)
    sigma_sum = (sigma.unsqueeze(-1) ** 2 + sigma.unsqueeze(-2) ** 2).clamp(min=1e-8).sqrt()
    d = mu_diff / sigma_sum
    phi_d = torch.exp(-0.5 * d * d) / (SQRT2 * SQRTPI)
    Phi_d = 0.5 * (1.0 + torch.erf(d / SQRT2))
    e_abs_xk_xl = sigma_sum * (2.0 * phi_d + d * (2.0 * Phi_d - 1.0))
    t2 = (pi.unsqueeze(-1) * pi.unsqueeze(-2) * e_abs_xk_xl).sum(dim=(-1, -2))
    return t1 - 0.5 * t2


def mdn_sample(pi, mu, sigma, n_samples=50):
    """Sample the mixture (hardened against NaN -> CUDA assert)."""
    pi = torch.nan_to_num(pi, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
    row = pi.sum(dim=-1, keepdim=True)
    bad = ~torch.isfinite(row) | (row <= 1e-8)
    if bad.any(): pi = torch.where(bad, torch.ones_like(pi), pi)
    pi = pi / pi.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = torch.nan_to_num(sigma, nan=1e-3, posinf=1e3, neginf=1e-3).clamp(min=1e-6)
    cat_idx = torch.multinomial(pi, n_samples, replacement=True)
    return mu.gather(1, cat_idx) + sigma.gather(1, cat_idx) * torch.randn_like(mu.gather(1, cat_idx))


print("SolarSDE-Rollout architecture loaded "
      "(latent neural SDE rolled forward via Euler-Maruyama + decoder; "
      "drop-in (pi, mean, std) interface).")
'''


# ============================================================
# ABLATIONS_ROLLOUT_CODE — ablations of the rollout architecture
# A1 full | A2 no-CTI diffusion | A4 no-persistence | A5 deterministic (no diffusion) | A7 no-covariates
# ============================================================
ABLATIONS_ROLLOUT_CODE = '''\
# ==== STAGE B (rollout): Ablations of RolloutLatentSDE components ====
MDN_CKPT = CHECKPOINT_DIR / "mdn_v2_best.pt"
STAGE_B_OUT = RESULTS_DIR / "ablation_results.csv"
if STAGE_B_OUT.exists():
    print(f"[SKIP] Ablations already done: {STAGE_B_OUT}")
    abl = pd.read_csv(STAGE_B_OUT)
elif not MDN_CKPT.exists():
    raise RuntimeError("ABLATIONS_ROLLOUT requires mdn_v2_best.pt from STAGE 0")
else:
    print("=" * 70); print("STAGE B (rollout): Ablations of RolloutLatentSDE"); print("=" * 70)
    SEQ_LEN_ABL = int(globals().get("SEQ_LEN", 16))
    N_EVAL_ABL  = min(800, len(data["test"]["Z"]) - 200)

    def _load_roll():
        m = RolloutLatentSDE(z_dim=Z_DIM, c_dim=C_DIM, n_components=3, seq_len=SEQ_LEN_ABL,
                             d_model=128, n_heads=4, n_layers=2, n_horizons=len(HORIZON_MIN)).to(DEVICE)
        m.load_state_dict(torch.load(MDN_CKPT, map_location=DEVICE, weights_only=False)); m.eval()
        return m

    def _eval_roll(model, zero_cov=False, tag=""):
        te = data["test"]; rows = []
        for h in HORIZONS:
            hm = HORIZON_MIN[h]
            idxs = list(range(SEQ_LEN_ABL - 1, min(SEQ_LEN_ABL - 1 + N_EVAL_ABL, len(te["Z"]) - h - 1)))
            yt_l, ys_l, rm_l = [], [], []
            for k in range(0, len(idxs), 32):
                ch = idxs[k:k+32]; B = len(ch)
                z_seq = np.stack([te["Z"][i-SEQ_LEN_ABL+1:i+1] for i in ch]).astype(np.float32)
                kt_seq = np.stack([te["kt"][i-SEQ_LEN_ABL+1:i+1] for i in ch]).astype(np.float32)
                if zero_cov or te["cov"].shape[1] == 0:
                    c_seq = np.zeros((B, SEQ_LEN_ABL, C_DIM), np.float32)
                else:
                    c_seq = np.stack([te["cov"][i-SEQ_LEN_ABL+1:i+1] for i in ch]).astype(np.float32)
                cti = np.array([te["cti"][i] for i in ch], np.float32)[:, None]
                kt_t = np.array([te["kt"][i] for i in ch], np.float32)
                gcs_t = np.array([te["gcs"][i+h] for i in ch], np.float32)
                hn = np.full((B, 1), h/180.0, np.float32)
                with torch.no_grad():
                    pi, mu, sd = model(torch.from_numpy(z_seq).to(DEVICE), torch.from_numpy(kt_seq).to(DEVICE),
                                       torch.from_numpy(c_seq).to(DEVICE), torch.from_numpy(cti).to(DEVICE),
                                       torch.from_numpy(hn).to(DEVICE))
                    ds = mdn_sample(pi, mu, sd, n_samples=N_SAMPLES).cpu().numpy()
                ghi = np.clip(kt_t[:, None] + ds, 0, 1.5) * gcs_t[:, None]
                for ii, i in enumerate(ch):
                    yt_l.append(te["ghi"][i+h]); ys_l.append(ghi[ii]); rm_l.append(bool(te["ramp"][i+h]))
            yt = np.array(yt_l, np.float32); ys = np.array(ys_l, np.float32); rm = np.array(rm_l, bool)
            m = all_metrics(yt, ys, is_ramp=rm); m["horizon_min"] = hm; m["n_eval"] = len(yt); m["ablation"] = tag
            rows.append(m)
        return pd.DataFrame(rows)

    parts = []
    print("\\n  A1: full rollout model")
    parts.append(_eval_roll(_load_roll(), tag="A1_full"))
    print("  A2: no CTI-gated diffusion")
    m2 = _load_roll()
    with torch.no_grad():
        for p in m2.diff_cti.parameters(): p.zero_()
        m2.sigma_pers_cti_alpha.fill_(-10.0)
    parts.append(_eval_roll(m2, tag="A2_no_cti"))
    print("  A4: no persistence-blend (w forced to 1)")
    m4 = _load_roll()
    with torch.no_grad(): m4.w_max_table.fill_(1.0); m4.w_max.fill_(1.0); m4.head_w.bias.fill_(10.0)
    parts.append(_eval_roll(m4, tag="A4_no_persistence"))
    print("  A5: deterministic rollout (no diffusion noise -> Neural ODE)")
    m5 = _load_roll()
    with torch.no_grad():
        for p in m5.diff_out.parameters(): p.zero_()
        m5.diff_out[0].bias.fill_(-30.0)   # softplus(-30) ~= 0 -> zero diffusion
    parts.append(_eval_roll(m5, tag="A5_no_sde_ODE"))
    print("  A7: no covariates")
    parts.append(_eval_roll(_load_roll(), zero_cov=True, tag="A7_no_covariates"))

    abl = pd.concat(parts, ignore_index=True); abl.to_csv(STAGE_B_OUT, index=False)
    print("\\nAblation summary @ h=10min:")
    print(abl[abl.horizon_min == 10][["ablation","crps","picp","pinaw","rmse"]].round(3).to_string(index=False))
    print(f"  -> saved {STAGE_B_OUT}")
'''
