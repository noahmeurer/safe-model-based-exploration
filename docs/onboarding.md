# Onboarding Guide: Safe Model-Based Exploration (OPAX & ActSafe)

This guide provides a comprehensive overview of the codebase architecture, control flow, and practical instructions for running and evaluating experiments. It reflects the core Model Predictive Control (MPC) loop, the Cross-Entropy Method (CEM) optimizer, and the Gaussian Process (GP) dynamics model.

---

## 1. Architecture & Control Flow

The codebase is structured around a Model Predictive Control (MPC) loop. The agent plans a full trajectory using a hallucinated GP model, executes only the first action in the true environment, and then replans.

### A. The Entry Point (`experiments/.../experiment.py`)
The experiment script wires together the environment, the GP model (`GPStatisticalModel`), the optimizer parameters (`iCemParams`), and the reward functions. It instantiates the central orchestrator (`ActSafeAgent` or `SafeHUCRL`) and calls `agent.run_episodes()`.

### B. The Main Loop (`smbrl/agent/actsafe.py`)
`SafeModelBasedAgent` is the core orchestrator. Its `run_episodes` method loops over episodes, calling `do_episode()` each time. 

A single `do_episode()` iteration consists of:
1. **Dynamics Training:** `train_dynamics_model()` updates the GP model using all historical data collected so far.
2. **Active Exploration (`simulate_on_true_env`):** The agent interacts with the true environment to collect new data.
   - It uses the `ExplorationReward` (intrinsic reward based on GP uncertainty).
   - **The MPC Loop:** For each step in the episode, it calls `optimizer.act()` to plan a 20-step trajectory, but only executes the **first action**.
   - **Action Repeat (Frame Skip):** The chosen action is repeated for `action_repeat` (e.g., 2) ticks in the true simulator to ease the burden on the GP model and extend the effective planning horizon.
3. **Zero-Shot Evaluation (`test_a_task`):** The agent temporarily stops exploring and evaluates its current GP model on specific downstream tasks (e.g., "Swing up", "Balance").
   - It swaps the intrinsic reward for the task's extrinsic reward.
   - It runs an MPC loop to maximize this extrinsic reward.
   - **Crucial:** The data collected during evaluation is saved to disk (`task_outputs.pkl`) but is **never added** to the training dataset.
4. **Data Aggregation:** The exploration data is appended to the global `Data` buffer to be used in the next episode's GP training phase.

### C. The Optimizer (`smbrl/optimizer/icem.py`)
When `optimizer.act()` is called, it triggers the Improved Cross-Entropy Method (iCEM):
1. **Sampling:** It samples `num_samples` (e.g., 500) action sequences of length `horizon` using colored noise.
2. **Evaluation:** For each candidate action sequence, it simulates `num_particles` (e.g., 10) parallel imagined rollouts through the learned system (`objective` -> `rollout_actions`). Each particle gets a distinct PRNG key, producing different stochastic transition realizations.
3. **Optimistic/Pessimistic aggregation:** Reward aggregation is `max` when `use_optimism=True` (else `mean`), approximating an optimistic inner selection over sampled plausible dynamics; cost aggregation is `max` when `use_pessimism=True` (else `mean`).
4. **Refinement (`step` function):** It sorts the sequences by their aggregated score, takes the top `num_elites` (e.g., 50), and updates the sampling distribution's mean and variance.
5. **Iteration:** This `step` function is executed `num_steps` (e.g., 5) times via a compiled `jax.lax.scan` loop.
6. **Return:** It extracts the absolute best sequence found across all iterations and returns **only the first action** (`self.best_sequence[0]`) to the MPC loop. The rest of the sequence is saved to "warm start" the next planning cycle.

### D. Dynamics & Intrinsic Reward (`smbrl/model_based_rl/active_exploration_system.py`)
During iCEM's simulated rollouts, `ExplorationDynamics.next_state()` is called:
- It queries the model for `pred.mean`, `pred.epistemic_std`, and `pred.aleatoric_std`.
- **Optimism:** It forms the transition mean as `pred.mean + beta * pred.epistemic_std * Normal(0, I)` (or delta form when `predict_difference=True`), so eta-like perturbations are sampled implicitly rather than optimized as explicit CEM variables.
- **Aleatoric sampling:** The rollout transition is sampled from `Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward)`, so aleatoric uncertainty enters through the distribution `scale`.
- **Intrinsic Reward:** It computes $r_t^{int}=\sum_j \log(1 + (\sigma_{\mathrm{ep},j}/\sigma_{\mathrm{al},j})^2)$ (with clipping for numerical stability), appends it to the predicted next-state vector, and `ExplorationReward` extracts it as the scalar reward.

In short: CEM searches over actions (`num_samples`), while particle rollouts approximate optimism over plausible dynamics (`num_particles`) via sampled eta-like perturbations.

---

## 2. Visualization & Analysis

Because evaluation data (`test_a_task`) is isolated from training data, it is saved to disk as `task_outputs.pkl` for offline analysis.

### 2D Phase Portraits
You can visualize the zero-shot evaluation performance across episodes using scripts in the `analysis/` folder:
- `analysis/make_2d_plots_task_trajectories.py`: Loads `task_outputs.pkl` and plots the phase portrait (e.g., Angle vs. Angular Velocity) of the pendulum, colored by episode to show learning progress.
- `analysis/make_2d_plots_exploration_trajectories.py`: Plots the actual exploration trajectories used for training.

### Video Rendering
You can explicitly see the policy execute in the true environment by rendering the saved evaluation trajectories:
- `smbrl/playground/visualize_cartpole.py`: Loads a saved trajectory, forces the `dm_control` physics engine into those exact states step-by-step, renders the RGB frames, and uses `ffmpeg` to stitch them into an MP4 video. You can adapt this script for the Pendulum or Racecar environments.

---
---

## Legacy Notes (Previous Onboarding Draft)

1. **Entry Point:** Experiment Configurations

    - **Experiment scripts** wire together the environment with the GP model and pass them to the core exploration loop. *Note: When running these scripts, you must pass the argument `alg_name == 'OPAX'` to bypass the safety constraints, effectively running pure OPAX.*
        - `experiments/pendulum_gp_full_exp/experiment.py`
        - `experiments/cartpole_gp_full_exp/experiment.py`
        - `experiments/racecar_gp_full_exp/experiment.py`

2. **Core Loop:** OPAX Orchestration - Once launched, control is handed over to the active exploration modules.

    - **Episodic loop** orchestrated by `smbrl/agent/actsafe.py` (specifically the `SafeModelBasedAgent`):
        - Acts as the central MBRL orchestrator.
        - Queries the agent for an exploratory trajectory.
        - Executes the trajectory in the environment to collect data.
        - Aggregates the data and triggers the GP to retrain/update its posteriors.

    - **Uncertainty & Intrinsic Rewards:** `smbrl/model_based_rl/active_exploration_system.py`:
        - Defines `ExplorationDynamics` and `ExplorationReward`.
        - Converts the GP's epistemic uncertainty into the intrinsic reward signal that drives OPAX exploration.

    *(Note: `smbrl/model_based_rl/main.py` is an alternative orchestrator used by the `safe_opax` directories, but is not used in the `gp_full_exp` entry points.)*

3. **Continuous Hallucination:** H-UCRL - When the active exploration system needs to pick actions that maximize information gain, it invokes an optimizer. This is where the H-UCRL algorithm comes into play.

    - `smbrl/optimizer/icem.py`: If the experiment is set to use sampling-based optimization instead of gradients, this module invokes the Improved Cross-Entropy Method (ICEM). It samples thousands of action sequences and evaluates them against the hallucinated bounds.
    - `smbrl/optimizer/ipopt_optimizer.py`: Implements the H-UCRL theory. Uses IPOPT, a continuous gradient-based optimizer, to jointly optimize for actions and hallucinated states within the GP's confidence intervals (rather than just the actions `a` alone).

4. **Statistical Model:** Gaussian Processes - To hallucinate bounds, the optimizer requires predictive means and variances from a GP.

    - `smbrl/dynamics_models/gps.py`: Implements the Gaussian Process dynamics model.
        - The optimizers query this module continuously during planning to obtain predictive mean `mu(s, a)` and variance `sigma(s, a)`.
        - The predictive variance `sigma` provides the "intrinsic reward" signal OPAX uses to drive exploration.

5. **Evaluation:** Zero-Shot Transfer - The goal of OPAX is to train a global dynamics model capable of solving any downstream task without further environment interactions.

    - The online exploration loop periodically runs zero-shot evaluations using `test_a_task`, replacing the intrinsic reward with the downstream task reward.
    - `analysis/several_episodes_in_line_zero_shot.py`:
        - Run this script to visualize the zero-shot task performance across episodes by loading the saved `task_outputs.pkl`.

    - `smbrl/envs/pendulum.py` & `smbrl/envs/cartpole/rewards.py`:
        - Define specific downstream task rewards (e.g., penalties for pendulum fall) used in evaluation.

    - `analysis/make_2d_plots_exploration_trajectories.py`:
        - Generates visualizations of state-space coverage, verifying that OPAX's policy explores the entire relevant space compared to random policies.