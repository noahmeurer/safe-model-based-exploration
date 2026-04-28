import numpy as np
import os
import sys
import argparse
from smbrl.utils.experiment_utils import Logger, hash_dict
from bsm.utils import Data


def experiment(
        project_name: str = 'ActSafeTest',
        alg_name: str = 'ActSafe',
        entity_name: str = 'sukhijab',
        exp_hash: str = '42',
        seed: int = 0,
        num_particles: int = 10,
        num_samples: int = 500,
        alpha: float = 0.2,
        num_steps: int = 5,
        exponent: int = 2,
        lambda_constraint: float = 1e6,
        icem_horizon: int = 20,
        episode_length: int = 50,
        action_repeat: int = 2,
        max_radius: float = 2.5,
        num_training_steps: int = 1_000,
        use_optimism: bool = True,
        use_pessimism: bool = True,
        log_wandb: bool = True,
        logs_dir: str = 'runs',
        num_gpus: int = 0,
        function_norm: float = 1.0,
        num_elites: int = 50,
        beta: float = 3.0,
        use_precomputed_kernel_params: bool = False,
        use_function_norms: bool = False,
        num_offline_data: int = 10,
        violation_eps: float = 0.1,
):
    if num_gpus == 0:
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax.numpy as jnp
    import jax.random as jr
    import chex
    import wandb
    from smbrl.agent.actsafe import ActSafeAgent, SafeHUCRL, Task
    from flax import struct
    from distrax import Normal
    from typing import Tuple
    from optax import constant_schedule
    from mbpo.systems.rewards.base_rewards import Reward
    from smbrl.optimizer.icem import iCemParams
    from smbrl.envs.racecar import RCCar, ToleranceReward, decode_angles
    from smbrl.playground.racecar_icem import RadiusBoundBinary
    from bsm.statistical_model import GPStatisticalModel
    from smbrl.dynamics_models.gps import ARD
    import jax
    from mbrl.utils.offline_data import OfflineData
    jax.config.update("jax_enable_x64", True)

    MARGIN = 10
    env = RCCar(dt=0.03, margin_factor=MARGIN)
    configs = dict(
        alg_name=alg_name,
        seed=seed,
        num_particles=num_particles,
        num_samples=num_samples,
        alpha=alpha,
        num_steps=num_steps,
        exponent=exponent,
        lambda_constraint=lambda_constraint,
        icem_horizon=icem_horizon,
        episode_length=episode_length,
        action_repeat=action_repeat,
        max_radius=max_radius,
        num_training_steps=num_training_steps,
        use_optimism=use_optimism,
        use_pessimism=use_pessimism,
        num_gpus=num_gpus,
        function_norm=function_norm,
        num_elites=num_elites,
        violation_eps=violation_eps,
        beta=beta,
        use_precomputed_kernel_params=use_precomputed_kernel_params,
        use_function_norms=use_function_norms,
        num_offline_data=num_offline_data,
    )
    key = jr.PRNGKey(seed)

    precomputed_normalization_stats = None
    num_training_steps = constant_schedule(num_training_steps)

    if log_wandb:
        if os.environ.get('WANDB_PROJECT') is not None:
            project_name = os.environ.get('WANDB_PROJECT')
        if os.environ.get('WANDB_ENTITY') is not None:
            entity_name = os.environ.get('WANDB_ENTITY')
        wandb.init(project=project_name,
                   config=configs,
                   dir=logs_dir,
                   entity=entity_name,
                   )

    class RacCarOfflineData(OfflineData):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def _sample_states(self,
                           key: chex.PRNGKey,
                           num_samples: int):
            key_pos, key_angle, key_vel, key_obs = jr.split(key, 4)
            init_pos = jax.random.uniform(key_pos, shape=(num_samples, 2), minval=-3, maxval=3)
            init_theta = jax.random.uniform(key_angle, shape=(num_samples, 1), minval=-jnp.pi, maxval=jnp.pi)
            init_vel = jax.random.normal(key_vel, shape=(num_samples, 3))
            state = jnp.concatenate([init_pos, init_theta, init_vel], axis=-1)
            state = self.env._state_to_obs(state, rng_key=key_obs)
            return state

        def _sample_actions(self,
                            key: chex.PRNGKey,
                            num_samples: int):
            actions = jr.uniform(key, shape=(num_samples, 2), minval=-1, maxval=1)
            return actions

    offline_data_gen = RacCarOfflineData(env=env)
    if use_precomputed_kernel_params:
        # using dummy model to fit to data for getting kernel params
        dummy_model = GPStatisticalModel(
            kernel=ARD(input_dim=env.observation_size + env.action_size, length_scale=0.05),
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
            logging_wandb=log_wandb,
            beta=jnp.ones(shape=(env.observation_size,)) * beta,
            fixed_kernel_params=False,
            normalization_stats=None,
            num_training_steps=constant_schedule(2_500),
        )

        model_key, key = jr.split(key)

        init_model_state = dummy_model.init(key=model_key)

        offline_data_key, key = jr.split(key)
        offline_data = offline_data_gen.sample_transitions(key=offline_data_key,
                                                           num_samples=100)
        offline_data = Data(inputs=jnp.concatenate([offline_data.observation, offline_data.action], axis=-1),
                            outputs=offline_data.next_observation - offline_data.observation,
                            )
        print('model state before update: ', init_model_state)
        updated_model_state = dummy_model.update(stats_model_state=init_model_state, data=offline_data)
        print('model state after update: ', updated_model_state)

        precomputed_kernel_params = {
            'pseudo_length_scale': updated_model_state.model_state.params['pseudo_length_scale']}
        precomputed_normalization_stats = updated_model_state.model_state.data_stats
        del dummy_model, init_model_state, updated_model_state, offline_data
        num_training_steps = constant_schedule(0)

    offline_data_key, key = jr.split(key, 2)
    offline_data = offline_data_gen.sample_transitions(key=offline_data_key,
                                                       num_samples=num_offline_data)
    offline_data = Data(inputs=jnp.concatenate([offline_data.observation, offline_data.action], axis=-1),
                        outputs=offline_data.next_observation - offline_data.observation,
                        )

    model = GPStatisticalModel(
        kernel=ARD(input_dim=env.observation_size + env.action_size, length_scale=0.05),
        input_dim=env.observation_size + env.action_size,
        output_dim=env.observation_size,
        output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
        logging_wandb=log_wandb,
        beta=jnp.ones(shape=(env.observation_size,)) * beta,
        fixed_kernel_params=use_precomputed_kernel_params,
        normalization_stats=precomputed_normalization_stats,
        num_training_steps=num_training_steps,
    )

    @chex.dataclass
    class RaceCarRewardParams:
        ctrl_cost_weight: chex.Array = struct.field(default_factory=lambda: jnp.array(0.005))
        goal: chex.Array = struct.field(default_factory=lambda: jnp.zeros(2))
        target_angle: chex.Array = struct.field(default_factory=lambda: jnp.zeros(1))

    class RCCarEnvReward(Reward):
        _angle_idx: int = 2
        dim_action: Tuple[int] = (2,)
        encode_angle: bool = True

        def __init__(self, ctrl_cost_weight: float = 0.005, bound: float = 0.1, margin_factor: float = MARGIN,
                     target_angle: float = 0.0,
                     ):
            super().__init__(x_dim=6 + self.encode_angle, u_dim=self.dim_action[0])
            self.ctrl_cost_weight = ctrl_cost_weight
            # Margin 20 seems to work even better (maybe try at some point)
            self.tolerance_reward = ToleranceReward(bounds=(0.0, bound), margin=margin_factor * bound,
                                                    value_at_margin=0.1, sigmoid='long_tail')
            self.target_angle = target_angle

        def forward(self, obs: jnp.array, action: jnp.array, next_obs: jnp.array, params: RaceCarRewardParams):
            """ Computes the reward for the given transition """
            reward_ctrl = self.action_reward(action)
            reward_state = self.state_reward(obs, next_obs, params)
            reward = reward_state + params.ctrl_cost_weight * reward_ctrl
            return reward

        @staticmethod
        def action_reward(action: jnp.array) -> jnp.array:
            """ Computes the reward/penalty for the given action """
            return - (action ** 2).sum(-1)

        def state_reward(self, obs: jnp.array, next_obs: jnp.array, params: RaceCarRewardParams) -> jnp.array:
            """ Computes the reward for the given observations """
            if self.encode_angle:
                next_obs = decode_angles(next_obs, angle_idx=self._angle_idx)
            pos_diff = next_obs[:2] - params.goal
            theta_diff = next_obs[2] - params.target_angle
            pos_dist = jnp.sqrt(jnp.sum(jnp.square(pos_diff), axis=-1))
            theta_dist = jnp.abs(((theta_diff + jnp.pi) % (2 * jnp.pi)) - jnp.pi)
            total_dist = jnp.sqrt(pos_dist ** 2 + theta_dist ** 2)
            reward = self.tolerance_reward(total_dist)
            return reward

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: RaceCarRewardParams,
                     x_next: chex.Array | None = None):
            chex.assert_shape(x, (self.x_dim,))
            chex.assert_shape(u, (self.u_dim,))
            chex.assert_shape(x_next, (self.x_dim,))
            reward = self.forward(
                obs=x,
                action=u,
                next_obs=x_next,
                params=reward_params,
            )
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> RaceCarRewardParams:
            default_reward_params = RaceCarRewardParams()
            return default_reward_params.replace(target_angle=self.target_angle)

    if alg_name == 'SafeHUCRL':
        alg = SafeHUCRL
    elif alg_name == 'ActSafe':
        alg = ActSafeAgent
    elif alg_name == 'HUCRL':
        alg = SafeHUCRL
        lambda_constraint = 0.0
    elif alg_name == 'OPAX':
        alg = ActSafeAgent
        lambda_constraint = 0.0
    else:
        raise NotImplementedError

    icem_params = iCemParams(
        num_particles=num_particles,
        num_samples=num_samples,
        alpha=alpha,
        num_steps=num_steps,
        exponent=exponent,
        lambda_constraint=lambda_constraint,
    )

    cost_fn = RadiusBoundBinary(horizon=icem_horizon,
                                max_radius=max_radius,
                                violation_eps=violation_eps, )

    agent = alg(
        env=env,
        model=model,
        episode_length=episode_length,
        action_repeat=action_repeat,
        cost_fn=cost_fn,
        test_tasks=[Task(reward=RCCarEnvReward(target_angle=0.0), name='Park front', env=env),
                    Task(reward=RCCarEnvReward(target_angle=jnp.pi), name='Park reverse', env=env),
                    ],
        predict_difference=True,
        num_training_steps=num_training_steps,
        icem_horizon=icem_horizon,
        icem_params=icem_params,
        log_to_wandb=log_wandb,
        use_pessimism=use_pessimism,
        use_optimism=use_optimism,
    )

    model_state = model.init(jr.PRNGKey(seed))

    # Here we set the model_state of the GP to the right one, we must ensure that we don't do
    # any GP training afterward
    if use_precomputed_kernel_params:
        model_state.model_state.params = precomputed_kernel_params
        model_state.model_state.data_stats = precomputed_normalization_stats

    env_name = 'racecar'
    agent.run_episodes(num_episodes=10,
                       key=key,
                       model_state=model_state,
                       folder_name=f'{logs_dir}/{alg_name}/{env_name}/{exp_hash}/',
                       data=offline_data,
                       )

    wandb.finish()


def main(args):
    """"""
    from pprint import pprint
    print(args)
    """ generate experiment hash and set up redirect of output streams """
    exp_hash = hash_dict(args.__dict__)
    if args.exp_result_folder is not None:
        os.makedirs(args.exp_result_folder, exist_ok=True)
        log_file_path = os.path.join(args.exp_result_folder, '%s.log ' % exp_hash)
        logger = Logger(log_file_path)
        sys.stdout = logger
        sys.stderr = logger

    pprint(args.__dict__)
    print('\n ------------------------------------ \n')

    """ Experiment core """
    np.random.seed(args.seed)

    experiment(
        project_name=args.project_name,
        entity_name=args.entity_name,
        alg_name=args.alg_name,
        action_repeat=args.action_repeat,
        num_particles=args.num_particles,
        num_samples=args.num_samples,
        alpha=args.alpha,
        num_steps=args.num_steps,
        exponent=args.exponent,
        lambda_constraint=args.lambda_constraint,
        icem_horizon=args.icem_horizon,
        episode_length=args.episode_length,
        max_radius=args.max_radius,
        num_training_steps=args.num_training_steps,
        use_optimism=bool(args.use_optimism),
        use_pessimism=bool(args.use_pessimism),
        log_wandb=bool(args.log_wandb),
        seed=args.seed,
        logs_dir=args.logs_dir,
        num_gpus=args.num_gpus,
        exp_hash=exp_hash,
        function_norm=args.function_norm,
        num_elites=args.num_elites,
        beta=args.beta,
        use_precomputed_kernel_params=bool(args.use_precomputed_kernel_params),
        use_function_norms=bool(args.use_function_norms),
        num_offline_data=args.num_offline_data,
        violation_eps=args.violation_eps,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MTTest')

    # general experiment args
    parser.add_argument('--logs_dir', type=str, default='logs')
    parser.add_argument('--project_name', type=str, default='ActSafeTest')
    parser.add_argument('--alg_name', type=str, default='ActSafe')
    parser.add_argument('--entity_name', type=str, default='sukhijab')
    parser.add_argument('--num_particles', type=int, default=10)
    parser.add_argument('--num_samples', type=int, default=500)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--num_steps', type=int, default=10)
    parser.add_argument('--exponent', type=float, default=1.0)
    parser.add_argument('--lambda_constraint', type=float, default=1e9)
    parser.add_argument('--icem_horizon', type=int, default=50)
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--max_radius', type=float, default=2.5)
    parser.add_argument('--num_training_steps', type=int, default=1_000)
    parser.add_argument('--use_optimism', type=int, default=1)
    parser.add_argument('--use_pessimism', type=int, default=1)
    parser.add_argument('--log_wandb', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--function_norm', type=float, default=1.0)
    parser.add_argument('--num_elites', type=int, default=50)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--use_precomputed_kernel_params', type=int, default=0)
    parser.add_argument('--use_function_norms', type=int, default=0)
    parser.add_argument('--num_offline_data', type=int, default=10)
    parser.add_argument('--violation_eps', type=float, default=0.5)

    parser.add_argument('--seed', type=int, default=1)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
