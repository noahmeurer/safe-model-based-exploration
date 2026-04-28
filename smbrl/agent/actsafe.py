import copy
import os.path
import pickle
from typing import Tuple, NamedTuple, List

import chex
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree as jt
import wandb
from brax.envs import Env as BraxEnv
from brax.envs import State
from brax.training.types import Metrics
from bsm.statistical_model import StatisticalModel, GPStatisticalModel
from bsm.utils.normalization import Data
from bsm.utils.type_aliases import ModelState
from distrax import Distribution, Normal
from flax import struct
from jaxtyping import Key, Array, PyTree, Float
from mbpo.systems.rewards.base_rewards import Reward, RewardParams
from optax import Schedule, constant_schedule

from smbrl.model_based_rl.active_exploration_system import ExplorationSystem, ExplorationReward, ExplorationDynamics
from smbrl.optimizer.icem import iCemParams, iCemTO, AbstractCost
# from smbrl.optimizer.ipopt_optimizer import IPOPTOptimizer, IPOPTParams
from smbrl.utils.utils import create_folder, ExplorationTrajectory


class Task(NamedTuple):
    reward: Reward
    name: str
    env: BraxEnv


class SafeModelBasedAgent:
    def __init__(self,
                 env: BraxEnv,
                 model: StatisticalModel,
                 episode_length: int,
                 action_repeat: int,
                 cost_fn: AbstractCost,
                 test_tasks: List[Task],
                 predict_difference: bool = True,
                 num_training_steps: Schedule = constant_schedule(1000),
                 icem_horizon: int = 20,
                 icem_params: iCemParams = iCemParams(),
                 ipopt_params: iCemParams = iCemParams(),
                 # ipopt_params: IPOPTParams = IPOPTParams(),
                 saving_frequency: int = 5,
                 log_to_wandb: bool = False,
                 train_task_index: int = -1,
                 use_optimism: bool = True,
                 use_pessimism: bool = True,
                 optimizer: str = 'icem'  # can be 'icem' or 'ipopt'
                 ):
        assert train_task_index >= -1
        assert train_task_index <= len(test_tasks)
        self.env = env
        if isinstance(model, GPStatisticalModel):
            jax.config.update("jax_enable_x64", True)
        self.model = model
        self.episode_length = episode_length
        self.action_repeat = action_repeat
        self.cost_fn = cost_fn
        self.cost_fn_env = copy.deepcopy(cost_fn)
        self.cost_fn_env.horizon = self.episode_length
        if hasattr(self.cost_fn_env, 'violation_eps'):
            self.cost_fn_env.violation_eps = 0
        self.test_tasks = test_tasks

        self.use_optimism = use_optimism
        self.use_pessimism = use_pessimism

        self.train_task_index = train_task_index

        self.predict_difference = predict_difference
        self.num_training_steps = num_training_steps
        self.icem_horizon = icem_horizon
        self.icem_params = icem_params
        self.saving_frequency = saving_frequency
        self.log_to_wandb = log_to_wandb
        self.optimizer = optimizer
        self.ipopt_params = ipopt_params

    def train_dynamics_model(self,
                             model_state: ModelState,
                             data: Data,
                             episode_idx: int) -> ModelState:
        model_state = self.model.update(data=data,
                                        stats_model_state=model_state)
        return model_state

    def test_a_task(self,
                    model_state: ModelState,
                    key: Key[Array, '2'],
                    task: Task,
                    ) -> Tuple[State, Float[Array, '... action_dim'], Float[Array, 'episode_length 1'], Metrics]:
        exploration_dynamics = ExplorationDynamics(x_dim=self.env.observation_size,
                                                   u_dim=self.env.action_size,
                                                   model=self.model,
                                                   )
        # To get a pure mean dynamics evaluation for OPAX, set beta to 0
        is_opax = self.icem_params.lambda_constraint == 0.0
        beta = jnp.zeros_like(model_state.beta) if is_opax else model_state.beta
        eval_model_state = model_state.replace(beta=beta)

        learned_system = ExplorationSystem(
            dynamics=exploration_dynamics,
            reward=task.reward,
        )
        key, subkey = jr.split(key)

        if self.optimizer == 'icem':
            use_optimism = self.use_optimism if not is_opax else False
            eval_icem_params = self.icem_params._replace(num_particles=1) if is_opax else self.icem_params
            optimizer = iCemTO(
                horizon=self.icem_horizon,
                action_dim=self.env.action_size,
                key=subkey,
                opt_params=eval_icem_params,
                system=learned_system,
                cost_fn=self.cost_fn,
                use_optimism=use_optimism,
                use_pessimism=self.use_pessimism,
            )
        # elif self.optimizer == 'ipopt':
        #     optimizer = IPOPTOptimizer(
        #         horizon=self.icem_horizon,
        #         action_dim=self.env.action_size,
        #         key=subkey,
        #         opt_params=self.ipopt_params,
        #         system=learned_system,
        #         cost_fn=self.cost_fn,
        #         use_optimism=self.use_optimism,
        #         use_pessimism=self.use_pessimism,
        #     )

        key, subkey = jr.split(key)
        optimizer_state = optimizer.init(key=subkey)

        dynamics_params = optimizer_state.system_params.dynamics_params.replace(model_state=eval_model_state)
        system_params = optimizer_state.system_params.replace(dynamics_params=dynamics_params)
        optimizer_state = optimizer_state.replace(system_params=system_params)

        env_state = task.env.reset(rng=key)

        collected_states = [env_state]
        actions = []

        for i in range(self.episode_length):
            action, optimizer_state = optimizer.act(env_state.obs, optimizer_state)
            for _ in range(self.action_repeat):
                env_state = self.env.step(env_state, action)
            collected_states.append(env_state)
            actions.append(action)

        collected_states = jt.map(lambda *xs: jnp.stack(xs), *collected_states)
        actions = jt.map(lambda *xs: jnp.stack(xs), *actions)
        # get task reward
        state = collected_states.obs[:-1]
        next_state = collected_states.obs[1:]
        reward_params = system_params.reward_params
        rewards_dist, _ = jax.vmap(task.reward, in_axes=(0, 0, None, 0))(state, actions, reward_params, next_state)
        rewards = rewards_dist.mean()
        costs = self.cost_fn_env(state, actions)
        metrics = {f'total_reward_{task.name}': jnp.sum(rewards).item(), f'cost_{task.name}': costs.item()}
        return collected_states, actions, rewards, metrics

    def get_train_rewards(self) -> Reward:
        if self.train_task_index == -1:
            exploration_reward = ExplorationReward(x_dim=self.env.observation_size,
                                                   u_dim=self.env.action_size, )
            return exploration_reward
        else:
            return self.test_tasks[self.train_task_index].reward

    def get_train_env_state(self, rng: jax.Array) -> State:
        if self.train_task_index == -1:
            return self.env.reset(rng=rng)
        else:
            env = self.test_tasks[self.train_task_index].env
            return env.reset(rng=rng)

    def simulate_on_true_env(self,
                             model_state: ModelState,
                             key: Key[Array, '2'], ) -> Tuple[
        PyTree[Array, 'episode_length ...'], Float[Array, 'episode_length action_dim'], Float[
            Array, 'episode_length 1'], Float[
            Array, 'episode_length 1'], Float[Array, '1']]:
        reward = self.get_train_rewards()

        exploration_dynamics = ExplorationDynamics(x_dim=self.env.observation_size,
                                                   u_dim=self.env.action_size,
                                                   model=self.model,
                                                   )
        learned_system = ExplorationSystem(
            dynamics=exploration_dynamics,
            reward=reward,
        )
        key, subkey = jr.split(key)

        if self.optimizer == 'icem':
            optimizer = iCemTO(
                horizon=self.icem_horizon,
                action_dim=self.env.action_size,
                key=subkey,
                opt_params=self.icem_params,
                system=learned_system,
                cost_fn=self.cost_fn,
                use_optimism=self.use_optimism,
                use_pessimism=self.use_pessimism,
            )
        # elif self.optimizer == 'ipopt':
        #     optimizer = IPOPTOptimizer(
        #         horizon=self.icem_horizon,
        #         action_dim=self.env.action_size,
        #         key=subkey,
        #         opt_params=self.ipopt_params,
        #         system=learned_system,
        #         cost_fn=self.cost_fn,
        #         use_optimism=self.use_optimism,
        #         use_pessimism=self.use_pessimism,
        #     )

        key, subkey = jr.split(key)
        optimizer_state = optimizer.init(key=subkey)

        dynamics_params = optimizer_state.system_params.dynamics_params.replace(model_state=model_state)
        system_params = optimizer_state.system_params.replace(dynamics_params=dynamics_params)
        optimizer_state = optimizer_state.replace(system_params=system_params)

        env_state = self.get_train_env_state(rng=key)

        collected_states = [env_state]
        actions = []
        intrinsic_rewards = []
        extrinsic_rewards = []
        # TODO: Should implement treatment of done flags
        for i in range(self.episode_length):
            action, optimizer_state = optimizer.act(env_state.obs, optimizer_state)
            print(f'Step {i}: reward is {optimizer_state.best_reward}')
            for _ in range(self.action_repeat):
                env_state = self.env.step(env_state, action)
                extrinsic_rewards.append(env_state.reward)
            # Calculate extrinsic reward
            z = jnp.concatenate([env_state.obs, action])
            pred = self.model(z, model_state)
            epistemic_std, aleatoric_std = pred.epistemic_std, pred.aleatoric_std
            intrinsic_reward = learned_system.dynamics.get_intrinsic_reward(epistemic_std=epistemic_std,
                                                                            aleatoric_std=aleatoric_std)
            intrinsic_rewards.append(intrinsic_reward)
            collected_states.append(env_state)
            actions.append(action)

        collected_states = jt.map(lambda *xs: jnp.stack(xs), *collected_states)
        actions = jt.map(lambda *xs: jnp.stack(xs), *actions)
        intrinsic_rewards = jt.map(lambda *xs: jnp.stack(xs), *intrinsic_rewards)
        extrinsic_rewards = jt.map(lambda *xs: jnp.stack(xs), *extrinsic_rewards)
        costs = self.cost_fn_env(collected_states.obs[:-1], actions)
        return collected_states, actions, intrinsic_rewards, extrinsic_rewards, costs

    def from_collected_transitions_to_data(self,
                                           collected_states: PyTree[Array, 'episode_length ...'],
                                           actions: Float[Array, 'episode_length action_dim']) -> Data:
        # TODO: Isn't this wrong, if we have a done flag in collected_states?
        states = collected_states.obs[:-1]
        next_states = collected_states.obs[1:]
        inputs = jnp.concatenate([states, actions], axis=-1)
        if self.predict_difference:
            outputs = next_states - states
        else:
            outputs = next_states
        return Data(inputs=inputs, outputs=outputs)

    def do_episode(self,
                   model_state: ModelState,
                   episode_idx: int,
                   data: Data,
                   key: Key[Array, '2'],
                   save_agent: bool = True,
                   train_model: bool = True,
                   folder_name: str = 'experiment_2024'
                   ) -> (ModelState, Data):
        if train_model:
            # If we collected some data already then we train dynamics model and the policy
            print(f'Start of dynamics training')
            model_state = self.train_dynamics_model(model_state=model_state,
                                                    data=data,
                                                    episode_idx=episode_idx)

        # We collect new data with the current policy
        print(f'Start of data collection')
        exploration_states, exploration_actions, intrinsic_rewards, extrinsic_rewards, cost = self.simulate_on_true_env(
            model_state=model_state,
            key=key)

        # import matplotlib.pyplot as plt
        # plt.plot(exploration_states.obs)
        # plt.axhline(y=-1.5, color='r', linestyle='-')
        # plt.axhline(y=1.5, color='r', linestyle='-')
        # plt.show()

        if self.log_to_wandb:
            wandb.log({
                'episode_idx': episode_idx,
                'intrinsic_rewards': jnp.sum(intrinsic_rewards).item(),
                'extrinsic_rewards': jnp.sum(extrinsic_rewards).item(),
                'constraint_cost': cost.item()
            })

        task_outputs = []
        for task in self.test_tasks:
            task_output = self.test_a_task(model_state=model_state, key=key, task=task)
            task_metrics = task_output[-1]
            task_outputs.append(task_output[:-1])
            if self.log_to_wandb:
                task_metrics['episode_idx'] = episode_idx
                wandb.log(task_metrics)
            else:
                print(task_metrics)
            print(f'End of task {task.name} evaluation')

        new_data = self.from_collected_transitions_to_data(exploration_states, exploration_actions)
        data = Data(inputs=jnp.concatenate([data.inputs, new_data.inputs]),
                    outputs=jnp.concatenate([data.outputs, new_data.outputs]), )

        # We save everything with pickle
        folder_name = os.path.join(folder_name, f'episode_{episode_idx}')
        create_folder(folder_name)

        if (not self.log_to_wandb) and save_agent:
            # Saving data to a pickle file
            with open(os.path.join(folder_name, 'data.pkl'), 'wb') as file:
                pickle.dump(data, file)

            with open(os.path.join(folder_name, 'model_state.pkl'), 'wb') as file:
                pickle.dump(model_state, file)

            with open(os.path.join(folder_name, 'exploration_trajectory.pkl'), 'wb') as file:
                pickle.dump(ExplorationTrajectory(states=exploration_states, actions=exploration_actions,
                                                  intrinsic_rewards=intrinsic_rewards,
                                                  extrinsic_rewards=extrinsic_rewards), file)

            with open(os.path.join(folder_name, 'task_outputs.pkl'), 'wb') as file:
                pickle.dump(task_outputs, file)

        if self.log_to_wandb and save_agent:
            folder_name = os.path.join(wandb.run.dir, 'saved_data', f'episode_{episode_idx}')
            create_folder(folder_name)

            # Saving data to a pickle file
            with open(os.path.join(folder_name, 'data.pkl'), 'wb') as file:
                pickle.dump(data, file)
            wandb.save(os.path.join(folder_name, 'data.pkl'), wandb.run.dir)

            with open(os.path.join(folder_name, 'model_state.pkl'), 'wb') as file:
                pickle.dump(model_state, file)

            wandb.save(os.path.join(folder_name, 'model_state.pkl'), wandb.run.dir)

            with open(os.path.join(folder_name, 'exploration_trajectory.pkl'), 'wb') as file:
                pickle.dump(ExplorationTrajectory(states=exploration_states, actions=exploration_actions,
                                                  intrinsic_rewards=intrinsic_rewards,
                                                  extrinsic_rewards=extrinsic_rewards,
                                                  ), file)

            wandb.save(os.path.join(folder_name, 'exploration_trajectory.pkl'), wandb.run.dir)

            with open(os.path.join(folder_name, 'task_outputs.pkl'), 'wb') as file:
                pickle.dump(task_outputs, file)

            wandb.save(os.path.join(folder_name, 'task_outputs.pkl'), wandb.run.dir)

        return model_state, data

    def run_episodes(self,
                     num_episodes: int,
                     key: Key[Array, '2'] = jr.PRNGKey(0),
                     model_state: ModelState | None = None,
                     data: Data | None = None,
                     folder_name: str = 'experiment_2024') -> (ModelState, Data):
        create_folder(folder_name)
        train_model = True
        if data is None:
            data = Data(inputs=jnp.zeros(shape=(0, self.env.observation_size + self.env.action_size)),
                        outputs=jnp.zeros(shape=(0, self.env.observation_size)))
            train_model = False

        for episode_idx in range(num_episodes):
            key, subkey = jr.split(key)
            train_model = train_model or episode_idx > 0
            print(f'Starting with Episode {episode_idx}')
            save_agent = (episode_idx % self.saving_frequency == 0) or (episode_idx == num_episodes - 1)
            model_state, data = self.do_episode(model_state=model_state,
                                                episode_idx=episode_idx,
                                                data=data,
                                                key=subkey,
                                                train_model=train_model,
                                                save_agent=save_agent,
                                                folder_name=folder_name)
            print(f'End of Episode {episode_idx}')
        return model_state, data


class ActSafeAgent(SafeModelBasedAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(train_task_index=-1, *args, **kwargs)


class SafeHUCRL(SafeModelBasedAgent):
    def __init__(self, train_task_index: int = 0, *args, **kwargs):
        assert train_task_index >= 0
        super().__init__(train_task_index=train_task_index, *args, **kwargs)


if __name__ == '__main__':
    from smbrl.envs.pendulum import PendulumEnv
    from smbrl.playground.pendulum_icem import VelocityBound
    from smbrl.dynamics_models.gps import ARD
    import optax

    from mbrl.utils.offline_data import PendulumOfflineData

    # num_offline_data = 100
    offline_data_gen = PendulumOfflineData()
    key = jr.PRNGKey(0)
    offline_data = None
    # offline_data_key, key = jr.split(key)
    # offline_data = offline_data_gen.sample_transitions(key=offline_data_key,
    #                                                    num_samples=num_offline_data)
    #
    # offline_data = Data(inputs=jnp.concatenate([offline_data.observation, offline_data.action], axis=-1),
    #                     outputs=offline_data.next_observation,
    #                     )

    env = PendulumEnv()
    log_wandb = True

    model = GPStatisticalModel(
        kernel=ARD(input_dim=env.observation_size + env.action_size),
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
        logging_wandb=False,
        f_norm_bound=3 * jnp.ones(shape=(env.observation_size,)),
        beta=None,
        num_training_steps=optax.constant_schedule(1000)
    )

    # model = DeterministicEnsemble(
    #     features=(256, 256),
    #     num_particles=5,
    #     input_dim=env.observation_size + env.action_size,
    #     output_dim=env.observation_size,
    #     output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
    #     logging_wandb=log_wandb)

    icem_horizon = 20


    @chex.dataclass
    class PendulumRewardParams:
        control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.02))
        angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
        target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(0.0))


    class PendulumReward(Reward):
        def __init__(self, target_angle: float = 0.0):
            super().__init__(x_dim=3, u_dim=1)
            self.target_angle = jnp.array(target_angle)

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: PendulumRewardParams,
                     x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
            chex.assert_shape(x, (self.x_dim,))
            chex.assert_shape(u, (self.u_dim,))
            chex.assert_shape(x_next, (self.x_dim,))
            # get intrinsic reward out
            theta, omega = jnp.arctan2(x[1], x[0]), x[-1]
            target_angle = reward_params.target_angle
            diff_th = theta - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            reward = -(reward_params.angle_cost * diff_th ** 2 +
                       0.1 * omega ** 2) - reward_params.control_cost * u ** 2
            reward = reward.squeeze()
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> PendulumRewardParams:
            default_reward_params = PendulumRewardParams()
            return default_reward_params.replace(target_angle=self.target_angle)


    class PendulumEnvBalance(PendulumEnv):
        def reset(self,
                  rng: jax.Array) -> State:
            # set initial state to upright
            state = State(pipeline_state=None,
                          obs=jnp.array([1.0, 0.0, 0.0]),
                          reward=jnp.array(0.0),
                          done=jnp.array(0.0), )
            if self.add_process_noise:
                state.info['process_noise_key'] = rng
            return state


    icem_params = iCemParams(
        num_particles=10,
        num_samples=500,
        alpha=0.2,
        num_steps=5,
        exponent=2,
        lambda_constraint=1e6
    )

    agent = ActSafeAgent(
        env=PendulumEnv(),
        model=model,
        episode_length=50,
        action_repeat=2,
        # cost_fn=None,
        cost_fn=VelocityBound(horizon=icem_horizon,
                              max_abs_velocity=6.0 - 10 ** (-3),
                              violation_eps=1e-3, ),
        test_tasks=[Task(reward=PendulumReward(), name='Swing up', env=env),
                    Task(reward=PendulumReward(), name='Balance', env=PendulumEnvBalance()),
                    Task(reward=PendulumReward(target_angle=jnp.pi), name='Keep down', env=env),
                    ],
        predict_difference=True,
        num_training_steps=constant_schedule(1000),
        icem_horizon=icem_horizon,
        icem_params=icem_params,
        log_to_wandb=log_wandb,
    )

    model_state = model.init(jr.PRNGKey(0))
    if log_wandb:
        wandb.init(project='act safe test')
    agent.run_episodes(num_episodes=20,
                       key=key,
                       model_state=model_state,
                       folder_name='Cost30Aug2024',
                       data=offline_data,
                       )
