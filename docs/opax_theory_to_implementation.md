# OPAX Theory-to-Implementation (iCEM Path)

This section explains how the OPAX planning objective is instantiated in the `gp_full_exp` pipeline when using iCEM.

## 1) From OPAX's optimistic control problem to the implemented optimizer objective

In OPAX, planning is formulated as an optimistic objective over controls and uncertainty-realization terms:

$$
\max_{u_{0:T-1}} \max_{\eta_{0:T-1}} J(u, \eta)
$$

with optimistic model rollouts of the form

$$
\hat x_{t+1} = \mu(\hat x_t, u_t) + \beta\,\sigma_{\mathrm{epi}}(\hat x_t, u_t)\odot \eta_t + w_t.
$$

In this code path, iCEM explicitly optimizes only $u_{0:T-1}$. The inner optimistic part is approximated by particle sampling:

1. For each candidate action sequence $u$, run $P$ particle rollouts of learned dynamics.
2. In each particle and timestep, an eta-like epistemic perturbation is sampled implicitly as $\xi_t \sim \mathcal{N}(0, I)$, and the transition mean is shifted by $\beta \, \sigma_{epi}\odot \xi_t$ (rather than explicitly optimizing bounded $\eta_t \in [-1, 1]^{d_x}$).
3. Aggregate particle returns with an optimistic reducer (`max` when `use_optimism=True`; otherwise `mean`).

So the implemented inner approximation is:

$$
\max_{\eta} J(u, \eta) \approx \max_{p \in \{1,\dots,P\}} J\left(u, \eta^{(p)}\right)
$$

where `eta^(p)` is implicit in the sampled epistemic perturbations of particle `p`, not an explicitly optimized CEM variable.

The final iCEM score used to rank action sequences is:

$$
\mathrm{score}(u)=\mathrm{Agg}_{\mathrm{reward}}(\mathbf{R}(u))
-\lambda\,\mathrm{ReLU}\!\left(\mathrm{Agg}_{\mathrm{cost}}(\mathbf{C}(u))\right)
$$

with reward aggregation set by optimism and cost aggregation set by pessimism. In this experiment pipeline, setting `alg_name='OPAX'` sets `lambda_constraint=0`, so the ReLU safety-penalty term is disabled and planning becomes reward-only.

## 2) Noise model note (single practical sentence)

For `pendulum_gp_full_exp`, aleatoric noise is homoscedastic per output dimension (fixed `output_stds`, here equal across all three state dimensions), and is equivalent to additive Gaussian process noise $w_t$ under the standard diagonal-noise GP assumption on the modeled transition targets.

## 3) Why CEM is used for actions but not for $\eta$, and compute impact

The implementation spends optimization budget on $u_{0:T-1}$ (the physically executed variable) and approximates the inner optimistic $\eta$-selection with particle sampling. This avoids a nested optimizer over $\eta$, which would significantly increase MPC compute.

Using the default pendulum settings:

- `num_samples = 500`
- `num_steps = 5`
- `num_particles = 10`
- `icem_horizon = 20`

Current per-MPC-step transition-evaluation scale is approximately:

$$
500 \times 5 \times 10 \times 20 = 500{,}000
$$

model transition evaluations (order-of-magnitude).

If a second CEM loop over $\eta$ were added:

- with a similar inner budget (`500` samples, `5` steps), compute would be multiplied by about $2500\times$;
- with a smaller inner budget (`50` samples, `3` steps), compute would still be multiplied by about $150\times$.

This is the main reason the implementation uses Monte Carlo particle aggregation for the optimistic inner term while keeping full CEM for action-sequence optimization.

## OPAX Symbol Dictionary (Theory -> Implementation)

| Theory Symbol | Meaning in OPAX | Implementation in this repo |
|---|---|---|
| `x_t` | State at time `t` | `env_state.obs` and `State.obs` in `smbrl/agent/actsafe.py` and envs (e.g. `smbrl/envs/pendulum.py`) |
| `u_t` | Control action at time `t` | `action` from `iCemTO.act()` (`smbrl/optimizer/icem.py`) |
| `z_t=(x_t,u_t)` | State-action input to dynamics model | Concatenated in `ExplorationDynamics.next_state()` (`smbrl/model_based_rl/active_exploration_system.py`) |
| `f*` | True unknown dynamics | Real simulator transition `env.step(...)` (e.g. pendulum env) |
| `mu_{n-1}(z)` | Posterior mean dynamics model at episode `n` | `pred.mean` from `self.model(z, model_state)` in `ExplorationDynamics.next_state()` |
| `sigma_epi,n-1(z)` | Posterior epistemic std of dynamics model | `pred.epistemic_std` from GP posterior (`bsm` statistical model; used in `ExplorationDynamics`) |
| `sigma_ale(z)` | Aleatoric/process-noise scale | `pred.aleatoric_std`; for this GP setup it is fixed `output_stds` (homoscedastic per output dim) |
| `beta_n` | Confidence-width multiplier on epistemic term | `pred.statistical_model_state.beta`, configured in `experiments/pendulum_gp_full_exp/experiment.py` |
| `eta_t in [-1,1]^{d_x}` (theory) | Optimistic hallucination direction | No explicit optimized `eta`-variable in iCEM path; approximated via sampled epistemic perturbations and particle aggregation in `iCemTO` |
| `w_t` | Additive process noise | Implemented as aleatoric sampling in rollout transition distribution (`Normal(..., scale=aleatoric_std)`) |
| `r_t^int` | Intrinsic exploration reward | `get_intrinsic_reward()` in `smbrl/model_based_rl/active_exploration_system.py`: `sum_j log(1 + (sigma_epi,j/sigma_ale,j)^2)` (with clipping for stability) |
| `J_n` | Planning objective over horizon | iCEM objective in `smbrl/optimizer/icem.py`: aggregated reward over particles minus `lambda * ReLU(cost)` |
| `H` | MPC planning horizon | `icem_horizon` passed to `iCemTO(horizon=...)` |
| `P` | Number of trajectory particles | `num_particles` in `iCemParams` |
| `N` | Number of episodes | `run_episodes(num_episodes=...)` in `smbrl/agent/actsafe.py` |
| `D_{1:n-1}` | Transitions collected up to episode `n-1` | Built by `from_collected_transitions_to_data()` and accumulated in `do_episode()` in `smbrl/agent/actsafe.py` |
| GP posterior conditioning | Posterior at planning time conditioned on current dataset | `model_state` updated before rollout, then fixed during iCEM rollouts for that episode |

### Notes

- In the iCEM implementation, CEM optimizes action sequences $u_{0:H-1}$, not a separate $\eta$-sequence.
- Optimism is implemented via particle aggregation (`max` vs `mean`) and stochastic epistemic perturbations during imagined rollouts.
- During imagined rollouts, the model samples from `Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward)`, so aleatoric uncertainty enters via distribution sampling, while epistemic optimism enters via the $\beta\,\sigma_{epi}\odot \xi$ shift in `loc`.
- In `pendulum_gp_full_exp`, aleatoric noise is homoscedastic per output dimension and acts as additive Gaussian noise under the model's diagonal-noise assumption.
