# Safe model-based exploration

## OPAX Onboarding

### Relevant Files
1. **Entry Point:** Experiment Configurations
    
    1.1. **Launcher scripts** instantiate the environment, the GP model, the optimizer and the exploration system:
        
    - [experiments/pendulum_gp_full_exp/launcher.py](experiments/pendulum_gp_full_exp/launcher.py)
    - [experiments/cartpole_gp_full_exp/launcher.py](experiments/cartpole_gp_full_exp/launcher.py)
    - [experiments/racecar_gp_full_exp/launcher.py](experiments/racecar_gp_full_exp/launcher.py)
        
    1.2. **Experiment scripts** wire together the environment with the GP model and pass them to the core exploration loop. *Note: When running these scripts, you must pass the argument `alg_name == 'OPAX'` to bypass the safety constraints, effectively running pure OPAX.*
        
    - [experiments/pendulum_gp_full_exp/experiment.py](experiments/pendulum_gp_full_exp/experiment.py)
    - [experiments/cartpole_gp_full_exp/experiment.py](experiments/cartpole_gp_full_exp/experiment.py)
    - [experiments/racecar_gp_full_exp/experiment.py](experiments/racecar_gp_full_exp/experiment.py)

2. **Core Loop:** OPAX Orchestration - Once launched, control is handed over to the active exploration modules.

    - **Episodic loop** orchestrated by [smbrl/agent/actsafe.py](smbrl/agent/actsafe.py) (specifically the `SafeModelBasedAgent`):
        - Acts as the central MBRL orchestrator.
        - Queries the agent for an exploratory trajectory.
        - Executes the trajectory in the environment to collect data.
        - Aggregates the data and triggers the GP to retrain/update its posteriors.

    - **Uncertainty & Intrinsic Rewards:** [smbrl/model_based_rl/active_exploration_system.py](smbrl/model_based_rl/active_exploration_system.py):
        - Defines `ExplorationDynamics` and `ExplorationReward`.
        - Converts the GP's epistemic uncertainty into the intrinsic reward signal that drives OPAX exploration.
        
    *(Note: `smbrl/model_based_rl/main.py` is an alternative orchestrator used by the `safe_opax` directories, but is not used in the `gp_full_exp` entry points.)*

3. **Continuous Hallucination:** H-UCRL - When the active exploration system needs to pick actions that maximize information gain, it invokes an optimizer. This is where the H-UCRL algorithm comes into play.

    - [smbrl/optimizer/ipopt_optimizer.py](smbrl/optimizer/ipopt_optimizer.py): Implements the H-UCRL theory. Uses IPOPT, a continuous gradient-based optimizer, to jointly optimize for actions and hallucinated states within the GP's confidence intervals (rather than just the actions $a$ alone).
    - [smbrl/optimizer/icem.py](smbrl/optimizer/icem.py): If the experiment is set to use sampling-based optimization instead of gradients, this module invokes the Improved Cross-Entropy Method (ICEM). It samples thousands of action sequences and evaluates them against the hallucinated bounds.

4. **Statistical Model:** Gaussian Processes - To hallucinate bounds, the optimizer requires predictive means and variances from a GP.
    
    - [smbrl/dynamics_models/gps.py](smbrl/dynamics_models/gps.py): Implements the Gaussian Process dynamics model. 
        - The optimizers query this module continuously during planning to obtain predictive mean $\mu(s, a)$ and variance $\sigma(s, a)$.
        - The predictive variance $\sigma$ provides the "intrinsic reward" signal OPAX uses to drive exploration.

5. **Evaluation:** Zero-Shot Transfer - The goal of OPAX is to train a global dynamics model capable of solving any downstream task without further environment interactions. After the exploration loop completes, you must evaluate the learned model:

    - [analysis/several_episodes_in_line_zero_shot.py](analysis/several_episodes_in_line_zero_shot.py): 
        - Run this script to test your trained GP model on a target task (e.g., Pendulum Swing-up).
        - Freezes the GP, swaps the objective from "maximize uncertainty" to "maximize task reward," and evaluates zero-shot task performance.

    - [smbrl/envs/pendulum.py](smbrl/envs/pendulum.py) & [smbrl/envs/cartpole/rewards.py](smbrl/envs/cartpole/rewards.py):
        - Define specific downstream task rewards (e.g., penalties for pendulum fall) used in evaluation.

    - [analysis/make_2d_plots_exploration_trajectories.py](analysis/make_2d_plots_exploration_trajectories.py):
        - Generates visualizations of state-space coverage, verifying that OPAX's policy explores the entire relevant space compared to random policies.


### Replication Checklist

## OPAX Theory-to-Implementation (iCEM Path)

This section explains how the OPAX planning objective is instantiated in the `gp_full_exp` pipeline when using iCEM.

### 1) From OPAX's optimistic control problem to the implemented optimizer objective

In OPAX, planning is formulated as an optimistic objective over controls and uncertainty-realization terms:

\[
\max_{u_{0:T-1}} \max_{\eta_{0:T-1}} J(u,\eta),
\]

with optimistic model rollouts of the form

\[
\hat x_{t+1} = \mu(\hat x_t,u_t) + \beta \sigma_{\text{epi}}(\hat x_t,u_t)\odot \eta_t + w_t.
\]

In this code path, iCEM explicitly optimizes only \(u_{0:T-1}\). The inner optimistic part is approximated by particle sampling:

1. For each candidate action sequence \(u\), run \(P\) particle rollouts of learned dynamics.
2. In each particle, epistemic perturbations are sampled in transition dynamics (Gaussian draws scaled by \(\beta \sigma_{\text{epi}}\)).
3. Aggregate particle returns with an optimistic reducer (`max` when `use_optimism=True`; otherwise `mean`).

So the implemented inner approximation is:

\[
\max_{\eta} J(u,\eta)\;\approx\;\max_{p\in\{1,\dots,P\}} J\big(u,\eta^{(p)}\big),
\]

where \(\eta^{(p)}\) is implicit in the sampled epistemic perturbations of particle \(p\), not an explicitly optimized CEM variable.

The final iCEM score used to rank action sequences is:

\[
\text{score}(u) = \text{Agg}_{\text{reward}}\!\left(\{R_p(u)\}_{p=1}^P\right)\;-\;\lambda\cdot\mathrm{ReLU}\!\left(\text{Agg}_{\text{cost}}\!\left(\{C_p(u)\}_{p=1}^P\right)\right),
\]

with reward aggregation set by optimism and cost aggregation set by pessimism.

### 2) Noise model note (single practical sentence)

For `pendulum_gp_full_exp`, aleatoric noise is homoscedastic per output dimension (fixed `output_stds`, here equal across all three state dimensions), and is equivalent to additive Gaussian process noise \(w_t\) under the standard diagonal-noise GP assumption on the modeled transition targets.

### 3) Why CEM is used for actions but not for \(\eta\), and compute impact

The implementation spends optimization budget on \(u_{0:T-1}\) (the physically executed variable) and approximates the inner optimistic \(\eta\)-selection with particle sampling. This avoids a nested optimizer over \(\eta\), which would significantly increase MPC compute.

Using the default pendulum settings:

- `num_samples = 500`
- `num_steps = 5`
- `num_particles = 10`
- `icem_horizon = 20`

Current per-MPC-step transition-evaluation scale is approximately:

\[
500 \times 5 \times 10 \times 20 = 500{,}000
\]

model transition evaluations (order-of-magnitude).

If a second CEM loop over \(\eta\) were added:

- with a similar inner budget (`500` samples, `5` steps), compute would be multiplied by about \(2500\times\);
- with a smaller inner budget (`50` samples, `3` steps), compute would still be multiplied by about \(150\times\).

This is the main reason the implementation uses Monte Carlo particle aggregation for the optimistic inner term while keeping full CEM for action-sequence optimization.

## OPAX Symbol Dictionary (Theory -> Implementation)

| Theory Symbol | Meaning in OPAX | Implementation in this repo |
|---|---|---|
| \(x_t\) | State at time \(t\) | `env_state.obs` and `State.obs` in `smbrl/agent/actsafe.py` and envs (e.g. `smbrl/envs/pendulum.py`) |
| \(u_t\) | Control action at time \(t\) | `action` from `iCemTO.act()` (`smbrl/optimizer/icem.py`) |
| \(z_t=(x_t,u_t)\) | State-action input to dynamics model | Concatenated in `ExplorationDynamics.next_state()` (`smbrl/model_based_rl/active_exploration_system.py`) |
| \(f^\*\) | True unknown dynamics | Real simulator transition `env.step(...)` (e.g. pendulum env) |
| \(\mu_{n-1}(z)\) | Posterior mean dynamics model at episode \(n\) | `pred.mean` from `self.model(z, model_state)` in `ExplorationDynamics.next_state()` |
| \(\sigma_{\text{epi},n-1}(z)\) | Posterior epistemic std of dynamics model | `pred.epistemic_std` from GP posterior (`bsm` statistical model; used in `ExplorationDynamics`) |
| \(\sigma_{\text{ale}}(z)\) | Aleatoric/process-noise scale | `pred.aleatoric_std`; for this GP setup it is fixed `output_stds` (homoscedastic per output dim) |
| \(\beta_n\) | Confidence-width multiplier on epistemic term | `pred.statistical_model_state.beta`, configured in `experiments/pendulum_gp_full_exp/experiment.py` |
| \(\eta_t\in[-1,1]^{d_x}\) (theory) | Optimistic hallucination direction | No explicit optimized \(\eta\)-variable in iCEM path; approximated via sampled epistemic perturbations and particle aggregation in `iCemTO` |
| \(w_t\) | Additive process noise | Implemented as aleatoric sampling in rollout transition distribution (`Normal(..., scale=aleatoric_std)`) |
| \(r_t^{\text{int}}\) | Intrinsic exploration reward | `get_intrinsic_reward()` in `smbrl/model_based_rl/active_exploration_system.py`: \(\sum_j \log(1 + (\sigma_{\text{epi},j}/\sigma_{\text{ale},j})^2)\) (with clipping for stability) |
| \(J_n\) | Planning objective over horizon | iCEM objective in `smbrl/optimizer/icem.py`: aggregated reward over particles minus \(\lambda\cdot\mathrm{ReLU}(\text{cost})\) |
| \(H\) | MPC planning horizon | `icem_horizon` passed to `iCemTO(horizon=...)` |
| \(P\) | Number of trajectory particles | `num_particles` in `iCemParams` |
| \(N\) | Number of episodes | `run_episodes(num_episodes=...)` in `smbrl/agent/actsafe.py` |
| Dataset \(D_{1:n-1}\) | Transitions collected up to episode \(n-1\) | Built by `from_collected_transitions_to_data()` and accumulated in `do_episode()` in `smbrl/agent/actsafe.py` |
| GP posterior conditioning | Posterior at planning time conditioned on current dataset | `model_state` updated before rollout, then fixed during iCEM rollouts for that episode |

### Notes

- In the iCEM implementation, CEM optimizes action sequences \(u_{0:H-1}\), not a separate \(\eta\)-sequence.
- Optimism is implemented via particle aggregation (`max` vs `mean`) and stochastic epistemic perturbations during imagined rollouts.
- In `pendulum_gp_full_exp`, aleatoric noise is homoscedastic per output dimension and acts as additive Gaussian noise under the model's diagonal-noise assumption.
