# Appendix A — Mathematical Formulation (in depth)

## A.1 Problem setup and notation

Let $P_t \ge 0$ denote rooftop PV power (kW) at minute $t$, and $x_t \in \mathbb{R}^{64\times64\times3}$ the co-located all-sky image. Define the **clear-sky-PV envelope** $C(m, \tau)$ as the empirical $0.92$ quantile of $P$ conditioned on calendar month $m$ and minute-of-day $\tau$, smoothed by a centered $\pm 15$-min rolling maximum within each month:
$$
C(m,\tau) = \max_{|\tau' - \tau| \le 15}\; Q_{0.92}\!\big(\{P_s : \text{month}(s)=m,\ \text{mod}(s)=\tau'\}\big).
$$
The **clear-sky index** is $k_t = \mathrm{clip}(P_t / C(m_t,\tau_t),\,0,\,1.3)$. Forecasting is performed on $k_t$ (a stationary, normalized target) and mapped back via $\hat P_{t+h} = \hat k_{t+h}\, C(m_{t+h},\tau_{t+h})$.

The forecast horizons are $h \in \mathcal{H}=\{1,5,10,15,20,30\}$ minutes. A history window of length $L=16$ minutes (matching the SkyGPT input) supplies $\{(z_s, k_s, c_s)\}_{s=t-L+1}^{t}$ where $z_s\in\mathbb{R}^{d_z}$ ($d_z=64$) is the cloud-state latent, $k_s$ the clear-sky index, and $c_s\in\mathbb{R}^{d_c}$ exogenous covariates.

## A.2 Cloud-state encoder and the Cloud Turbulence Index (CTI)

A convolutional VAE $E_\phi$ maps each frame to a posterior mean $z_t = \mu_\phi(x_t)\in\mathbb{R}^{64}$. The encoder is trained with the standard ELBO
$$
\mathcal{L}_{\text{VAE}} = \mathbb{E}_{q_\phi(z|x)}\!\left[\|x-\hat x\|_2^2\right] + \beta\, \mathrm{KL}\!\big(q_\phi(z|x)\,\|\,\mathcal{N}(0,I)\big),\qquad \beta=0.01 .
$$
The **Cloud Turbulence Index** is a deterministic functional of the latent trajectory: over a window of $W=10$ frames,
$$
\mathrm{CTI}_t = \Big\| \mathrm{Var}\big(\{\, z_{s} - z_{s-1} \,\}_{s=t-W+1}^{t}\big) \Big\|_2 ,
$$
the $\ell_2$-norm of the per-dimension variance of the latent velocity. It is large at cloud edges / broken cloud (rapidly changing latent) and near zero under uniform clear or overcast sky. CTI is robustly normalized by its training $90$th percentile, $\widetilde{\mathrm{CTI}}_t = \mathrm{clip}(\mathrm{CTI}_t / q_{0.9}^{\text{train}},\,0,\,10)$, giving it $O(1)$ dynamic range so it can drive the diffusion gate.

## A.3 Latent neural SDE: Mixture of Ornstein–Uhlenbeck processes

We model the forecast residual $\delta k_{t+h} = k_{t+h}-k_t$ (the deviation from smart-persistence) as the marginal of a **mixture of $K$ Ornstein–Uhlenbeck (OU) processes**. A transformer encoder $g_\theta$ ingests the $L$-step history and the conditioning $(\widetilde{\mathrm{CTI}}_t, h)$ to produce a context vector $\xi_t = g_\theta(\{(z_s,k_s,c_s)\}, \widetilde{\mathrm{CTI}}_t, h)\in\mathbb{R}^{128}$. From $\xi_t$, per-component heads emit mixing weights $\pi\in\Delta^{K-1}$, means $\mu\in\mathbb{R}^K$, mean-reversion rates $\theta\in\mathbb{R}_{>0}^K$, and base volatilities $\sigma_0\in\mathbb{R}_{>0}^K$.

Each component $k$ is the OU SDE
$$
\mathrm{d}Y^{(k)}_s = \theta_k\big(\mu_k - Y^{(k)}_s\big)\,\mathrm{d}s + \sigma_k\,\mathrm{d}W_s,\qquad Y^{(k)}_0 = 0 ,
$$
with the **CTI-gated diffusion**
$$
\sigma_k = \sigma_{0,k}\,\big(1 + \mathrm{Softplus}(\text{gate}_k(\widetilde{\mathrm{CTI}}_t))\big).
$$
This is the central physics-informed constraint: the stochastic forcing is amplified precisely when the cloud field is turbulent.

### Closed-form OU marginal

The OU process has a Gaussian marginal at any horizon $h$ in closed form. Solving the linear SDE,
$$
Y^{(k)}_h \sim \mathcal{N}\!\Big(\underbrace{\mu_k\big(1-e^{-\theta_k h}\big)}_{m_k(h)},\ \underbrace{\tfrac{\sigma_k^2}{2\theta_k}\big(1-e^{-2\theta_k h}\big)}_{v_k(h)}\Big).
$$
Thus the per-component predictive law for $\delta k_{t+h}$ is $\mathcal{N}(m_k(h), v_k(h))$ — **no autoregressive rollout and no path simulation are required**, which is the key computational distinction from generative video models.

### Persistence-anchored blended marginal

To guarantee the model degrades gracefully to a strong baseline at short horizons (where $k$ is $\sim 0.99$ autocorrelated), we blend the OU mixture with a persistence component $\mathcal{N}(0,\sigma_{\text{pers}}^2(h))$, where $\sigma_{\text{pers}}(h)$ is estimated from same-day clear-sky-index increments on the extended data. With a learnable blend weight $w = w_{\max}(h)\cdot \mathrm{sigmoid}(\text{head}_w(\xi_t)) \in[0, w_{\max}(h)]$, the full predictive density of $\delta k_{t+h}$ is the $(K{+}1)$-component Gaussian mixture
$$
p(\delta k_{t+h}\mid \cdot) = (1-w)\,\mathcal{N}\big(0,\ s_h^2\,\sigma_{\text{pers}}^2(h)\big) \;+\; w\sum_{k=1}^{K}\pi_k\,\mathcal{N}\big(m_k(h),\ s_h^2\,v_k(h)\big),
$$
where $s_h$ is the post-hoc conformal scale (Appendix A.5). The per-horizon cap $w_{\max}(h)$ increases with $h$ (persistence is near-optimal at $h=1$), preventing short-horizon over-confidence.

The PV forecast distribution is obtained by the affine map $\hat P_{t+h} = (k_t + \delta k_{t+h})\,C(m_{t+h},\tau_{t+h})$, sampled by $N$ Monte-Carlo draws from the mixture.

## A.4 Training objective: closed-form mixture CRPS

We train by minimizing the **Continuous Ranked Probability Score** of the blended $(K{+}1)$-component mixture against the realized $\delta k_{t+h}$, where the persistence atom $\mathcal{N}(0,\sigma_{\text{pers}}^2)$ is treated as component $0$ with weight $\pi_0 = 1-w$ and the OU atoms have weights $\pi_k = w\,\tilde\pi_k$ ($k=1,\dots,K$). All sums below run over the full index set $\{0,1,\dots,K\}$. For a single Gaussian $\mathcal{N}(\mu,\sigma)$ and observation $y$, CRPS has the closed form
$$
\mathrm{CRPS}\big(\mathcal{N}(\mu,\sigma),y\big) = \sigma\left[\,\frac{y-\mu}{\sigma}\big(2\Phi(\tfrac{y-\mu}{\sigma})-1\big) + 2\varphi(\tfrac{y-\mu}{\sigma}) - \tfrac{1}{\sqrt{\pi}}\,\right],
$$
with $\varphi,\Phi$ the standard normal pdf/cdf. For a mixture $\sum_k \pi_k \mathcal{N}(\mu_k,\sigma_k)$, CRPS admits the exact decomposition
$$
\mathrm{CRPS} = \sum_k \pi_k\, \mathbb{E}\,|X_k - y| \;-\; \tfrac{1}{2}\sum_{k}\sum_{l}\pi_k\pi_l\, \mathbb{E}\,|X_k - X_l|,
$$
where $\mathbb{E}|X_k-y| = \sigma_k\, A\!\big(\tfrac{y-\mu_k}{\sigma_k}\big)$ with $A(z)=2\varphi(z)+z(2\Phi(z)-1)$, and $\mathbb{E}|X_k - X_l| = \sqrt{\sigma_k^2+\sigma_l^2}\, A\!\big(\tfrac{\mu_k-\mu_l}{\sqrt{\sigma_k^2+\sigma_l^2}}\big)$ since $X_k-X_l\sim\mathcal{N}(\mu_k-\mu_l,\sigma_k^2+\sigma_l^2)$. Because the index set includes the persistence atom, the cross terms $\mathbb{E}|X_0-X_l|$ (persistence $\times$ OU) are present in the double sum, so the objective scores the *blended* law, not the OU mixture alone. It is **fully differentiable** through $\pi,\mu,\sigma$ and the blend weight $w$ — no sampling-based gradient estimator is needed, which is what lets the persistence-blend weight train stably.

**Relation of the training loss to the reported metric.** Training minimizes CRPS in clear-sky-index ($\delta k$) units. Evaluation reports CRPS in PV-power (kW). Since the map $\delta k \mapsto P=(k_t+\delta k)\,C(m_{t+h},\tau_{t+h})$ is affine with a per-sample positive scale $C$, CRPS transforms pointwise as $\mathrm{CRPS}_P = C\,\mathrm{CRPS}_{\delta k}$. The two objectives therefore differ by the heteroscedastic factor $C$; we report this explicitly because the model is trained on the unit-less score and evaluated on the kW score.

## A.5 Mondrian (group-conditional) conformal calibration

After training we calibrate coverage by **genuine split-conformal prediction restricted to a partition** (Mondrian conformal; Vovk et al., 2005). The partition is the grid of (horizon $h$, CTI-quartile $q$) cells. For a target level $1-\alpha$ we use the standardized absolute residual as the nonconformity score, $R_i = |y_i - \hat\mu_i| / \hat\sigma_i$, computed on the validation set. Within each cell of $n$ calibration points the conformal scale is the finite-sample-inflated empirical quantile
$$
s_{h,q}(\alpha) = \frac{\widehat{Q}_{\lceil (n+1)(1-\alpha)\rceil / n}\big(\{R_i\}_{i\in \text{cell}}\big)}{\Phi^{-1}(1-\alpha/2)},
$$
and the calibrated interval at a test point in that cell is $[\hat\mu \mp \Phi^{-1}(1-\alpha/2)\,s_{h,q}\,\hat\sigma]$. Under exchangeability within a cell, split-conformal guarantees the finite-sample marginal coverage bound
$$
\mathbb{P}\big(y_{\text{test}}\in [\hat\mu \mp \Phi^{-1}(1-\alpha/2)\,s_{h,q}\,\hat\sigma]\big) \ge 1-\alpha
$$
*conditional on the cell*, provided $n \ge \lceil (n+1)(1-\alpha)\rceil$; for cells with insufficient calibration samples (notably the high-CTI tail) the scale defaults to $1$ and the guarantee degrades gracefully to the model's nominal coverage. Crucially, the scale is a **quantile of held-out nonconformity scores — not a CRPS-minimizing search on validation** — so the reported test CRPS and coverage are computed on data untouched by calibration, avoiding metric double-dipping. We report test-set PICP at multiple nominal levels $\{0.5,0.7,0.8,0.9,0.95\}$ (Appendix B) to substantiate the coverage claim across the calibration curve, not at $90\%$ alone.

## A.6 Evaluation metrics (formal)

- **CRPS** (kW): $\mathrm{CRPS}(F,y)=\int_{\mathbb{R}}\big(F(u)-\mathbb{1}\{u\ge y\}\big)^2\mathrm{d}u$; estimated from $N$ samples by $\widehat{\mathrm{CRPS}}=\tfrac1N\sum_i|s_i-y|-\tfrac{1}{2N^2}\sum_{i,j}|s_i-s_j|$.
- **PICP** (coverage): fraction of observations within the central $90\%$ predictive interval; target $0.90$.
- **PINAW** (sharpness): mean $90\%$-PI width normalized by the observed range.
- **Winkler / interval score** at level $\alpha$: $W = (u-l) + \tfrac{2}{1-\alpha}\big[(l-y)\mathbb{1}\{y<l\} + (y-u)\mathbb{1}\{y>u\}\big]$.
- **Forecast skill**: $\mathrm{FS}=1-\mathrm{CRPS}_{\text{model}}/\mathrm{CRPS}_{\text{smart-pers}}$.
- **Diebold–Mariano**: $\mathrm{DM}=\bar d/\sqrt{\widehat{\mathrm{Var}}(\bar d)}$ on the CRPS loss differential $d_i$, asymptotically $\mathcal{N}(0,1)$.
- **PIT**: $u_i=F_i(y_i)$; calibration $\Leftrightarrow$ $\{u_i\}\sim\mathrm{Unif}[0,1]$, tested by Kolmogorov–Smirnov.
- **Reliability / ECE**: mean absolute gap between nominal and empirical coverage across levels $\{0.5,\dots,0.95\}$.
