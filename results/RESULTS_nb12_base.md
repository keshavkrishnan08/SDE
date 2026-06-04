# Fast-iterate run — ARCH=base, MOTION_GRID=1 (notebook 12)

Best closed-form result so far. CRPS-optimal calibration fully in effect.

## All-weather test (per-horizon)
| h | CRPS | RMSE | PICP | PINAW |
|---|---|---|---|---|
| 1  | 0.32 | 1.37 | 0.944 | 0.064 |
| 5  | 0.69 | 2.67 | 0.939 | 0.152 |
| 10 | 0.79 | 2.90 | 0.932 | 0.176 |
| 15 | 0.87 | 3.21 | 0.921 | 0.177 |
| 20 | 0.95 | 3.44 | 0.901 | 0.164 |
| 30 | 1.08 | 3.68 | 0.893 | 0.155 |
(Best yet — earlier closed-form h=15 was 1.03; calibration change improved sharpness.)

## SkyGPT cloudy test (identical Nov-Dec 2019)
| h | SolarSDE | smart-pers | skill |
|---|---|---|---|
| 1  | 1.237 | 1.241 | +0.4% |
| 5  | 2.154 | 2.150 | -0.1% |
| 10 | 2.690 | 2.748 | +2.1% |
| 15 | **3.026** | 3.105 | +2.5% |
| 20 | 3.303 | 3.290 | -0.4% |
| 30 | 3.842 | 3.471 | -10.7% |

Head-to-head h=15: SolarSDE 3.026 vs SkyGPT 2.810 (+7.7% behind, was +13%).
Now BEATS smart-persistence at h=1/10/15 (earlier it tied/lost).

## Sweep diagnostic
- val-selected (reportable): n=200, scale=0.85, blend=0.0 -> h15 = 3.036 (+8.0%)
- test-tuned oracle (NOT reportable): n=200, scale=1.15, blend=0.5 -> h15 = 2.886 (+2.7%)
  Oracle nearly beats 2.81 -> the structural gap is narrowing.

## Takeaway
Baseline config already at 3.03 with oracle 2.886. The spatial-motion (MOTION_GRID=3)
and bigger-head (bigmix) experiments — designed to target the cloudy gap — have NOT
been run yet and now have a real shot at crossing 2.81.
Next to try: ARCH=bigmix + MOTION_GRID=3.
