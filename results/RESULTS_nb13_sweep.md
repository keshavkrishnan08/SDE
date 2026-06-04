# ARCH x MOTION_GRID sweep (notebook 13) — NEGATIVE result for spatial motion

SkyGPT cloudy-test CRPS (kW), h=15 = the head-to-head horizon (SkyGPT pub = 2.810):

| config | h1 | h5 | h10 | h15 | h20 | h30 |
|---|---|---|---|---|---|---|
| base + grid1 (prior best, nb12) | 1.237 | 2.154 | 2.690 | **3.026** | 3.303 | 3.842 |
| base + grid3 | 1.269 | 2.327 | 3.133 | 3.630 | 4.005 | 4.193 |
| bigmix + grid3 | 1.240 | 2.188 | 2.797 | 3.113 | 3.388 | 3.687 |

## Findings
1. **Spatial motion grid (MOTION_GRID=3) HURTS.** base went 3.026 -> 3.630 at h=15
   (all-weather h=15 also 0.87 -> 1.08). The 27-dim grid-pooled optical flow on
   64x64 downsampled cloudy frames is noisy; the model overfits it on clear-sky
   training and generalizes worse on cloudy. Calibration destabilized (Q4
   multiplier hit 2.0). => Do NOT use MOTION_GRID>1.
2. **The bigmix head helps.** At grid3, bigmix (3.113) beat base (3.630) by 14%.
   The heavy-tail 8-component mixture is the right direction.
3. **Best so far remains base + grid1 = 3.026** (+7.7% vs SkyGPT).
4. **UNTESTED + promising: bigmix + grid1** (heavy-tail head WITHOUT the noisy
   spatial features). If bigmix helps the way it did at grid3, bigmix+grid1 could
   beat 3.026 and approach 2.81.

## Honest implication for the paper
Simple optical-flow spatial proxies do not extract usable cloud-position signal
from 64x64 frames — reinforces that closing the SkyGPT gap needs actual future-
frame prediction (a major rebuild), not cheap feature engineering. The drop-in
levers are nearly exhausted; bigmix+grid1 is the last cheap thing worth trying.

## Next experiment
SWEEP_CONFIGS = [("bigmix", 1), ("wide", 1), ("deep", 1)]  # head variants at grid1
