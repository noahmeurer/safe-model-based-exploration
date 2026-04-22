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
