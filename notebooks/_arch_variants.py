"""Architecture variants for the fast-iterate notebook (12) — for the SkyGPT
h= benchmark experiments only.

Five drop-in architectures the SkyGPT eval can test back-to-back. Four are the
existing closed-form Mixture-of-OU at different capacities (just different
constructor args — the model already takes n_components / d_model / n_layers);
one swaps the transformer encoder for a GRU. All share the exact
(pi, mean, std) interface, so STAGE_0 / SkyGPT eval / sweep run unchanged.

ARCH menu (set ARCH = "..." in notebook 12):
  "base"      K=3  d=128 L=2   — current closed-form (reference)
  "bigmix"    K=8  d=128 L=3   — heavy-tail mixture (8 OU components) for cloudy ramps
  "wide"      K=4  d=256 L=2   — wider transformer (more representation capacity)
  "deep"      K=4  d=128 L=4   — deeper transformer (longer-range history mixing)
  "gru"       K=3  d=128 L=2   — GRU temporal encoder (different temporal inductive bias)

HONEST NOTE: all five consume the same 64-d POOLED VAE latent, which discards
*where* clouds are. That pooling — not the forecasting head — is the main reason
the model only ties smart-persistence on cloudy days. So these variants test
capacity/encoder choices and may help at the margin, but the architecture that
would genuinely beat SkyGPT needs spatial/optical-flow cloud-motion in image
space (a larger rebuild). These are the cheap, fast things to try first.
"""

ARCH_VARIANTS_CODE = '''\
# ==== Architecture variants for the SkyGPT h= experiments ====
# GRU encoder variant: subclasses the closed-form model and swaps the transformer
# history encoder for a GRU. Everything else (Mixture-of-OU marginal, calibration
# buffers, persistence-blend) is inherited unchanged.
class GRULatentSDE(TemporalLatentSDE):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        d = self.d_model if hasattr(self, "d_model") else self.step_embed.out_features
        # replace the transformer with a 2-layer GRU over the same step embeddings
        try:
            del self.transformer
        except Exception:
            pass
        self.gru = nn.GRU(input_size=d, hidden_size=d, num_layers=2,
                          batch_first=True, dropout=0.1)

    def _encode(self, z_seq, kt_seq, c_seq, cti, h_norm):
        x = torch.cat([z_seq, kt_seq.unsqueeze(-1), c_seq], dim=-1)
        x = self.step_embed(x) + self.pos_embed.unsqueeze(0)
        out, _ = self.gru(x)
        last = out[:, -1, :]
        cti_h_in = torch.cat([cti, h_norm], dim=-1)
        return last + self.cti_h_embed(cti_h_in)

# d_model needs to be an attribute for the GRU variant; ensure it exists.
if not hasattr(TemporalLatentSDE, "_d_model_patched"):
    _orig_init = TemporalLatentSDE.__init__
    def _patched_init(self, *a, **k):
        _orig_init(self, *a, **k)
        self.d_model = k.get("d_model", a[4] if len(a) > 4 else 128)
    TemporalLatentSDE.__init__ = _patched_init
    TemporalLatentSDE._d_model_patched = True

# Variant menu: (n_components, d_model, n_layers, encoder_class)
ARCH_MENU = {
    "base":   (3, 128, 2, TemporalLatentSDE),
    "bigmix": (8, 128, 3, TemporalLatentSDE),
    "wide":   (4, 256, 2, TemporalLatentSDE),
    "deep":   (4, 128, 4, TemporalLatentSDE),
    "gru":    (3, 128, 2, GRULatentSDE),
}
_arch = str(globals().get("ARCH", "base")).lower()
if _arch not in ARCH_MENU:
    print(f"[WARN] unknown ARCH='{_arch}' — using 'base'"); _arch = "base"
_K, _D, _L, _CLS = ARCH_MENU[_arch]
ARCH_N_COMPONENTS = _K; ARCH_D_MODEL = _D; ARCH_N_LAYERS = _L; ARCH_N_HEADS = 4
# point the alias STAGE_0 / eval / sweep use at the selected encoder class
TemporalLatentSDE_SELECTED = _CLS
ClosedFormSDE = _CLS

# build_sde_model(seq_len): factory used by SkyGPT eval / sweep so they
# instantiate the SAME architecture that was trained (loads cleanly).
def build_sde_model(seq_len):
    return _CLS(z_dim=Z_DIM, c_dim=C_DIM, n_components=ARCH_N_COMPONENTS,
                seq_len=seq_len, d_model=ARCH_D_MODEL, n_heads=ARCH_N_HEADS,
                n_layers=ARCH_N_LAYERS, n_horizons=len(HORIZON_MIN)).to(DEVICE)

print(f"[ARCH] '{_arch}': n_components={_K} d_model={_D} n_layers={_L} encoder={_CLS.__name__}")
'''
