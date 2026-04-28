import argparse
import sys

import numpy as np

from smbrl.utils.experiment_utils import Logger, hash_dict


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
        max_position: float = 0.5,
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
        num_offline_data: int = 0,
        violation_eps: float = 0.1,
        optimizer: str = 'icem',
):
    if num_gpus == 0:
        import os
        os.environ['JAX_PLATFORMS'] = 'cpu'

    import jax.numpy as jnp
    import jax.random as jr
    import chex
    import wandb
    from smbrl.agent.actsafe import ActSafeAgent, SafeHUCRL
    from flax import struct
    from distrax import Distribution, Normal
    from typing import Tuple
    from optax import constant_schedule
    from mbpo.systems.rewards.base_rewards import Reward, RewardParams
    from smbrl.optimizer.icem import iCemParams
    from smbrl.envs.cartpole_lenart import CartPoleEnv, CartPoleOfflineData
    from smbrl.playground.cartpole_icem import PositionBoundBinary
    from bsm.statistical_model import GPStatisticalModel
    from smbrl.dynamics_models.gps import ARD
    from jaxtyping import Float, Array, Scalar
    from bsm.utils import Data, Stats, DataStats
    from smbrl.agent.actsafe import Task

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
        max_position=max_position,
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
        optimizer=optimizer
    )
    import jax
    jax.config.update("jax_enable_x64", True)

    precomputed_kernel_params = {
        'pseudo_length_scale': jnp.array([[7.16197382, 6.65598727, 1.27592871, 5.13755356, 4.53211409,
                                           9.09040351],
                                          [8.19072527, 2.16344693, 1.40387256, 9.7920551, 2.02477876,
                                           7.42569151],
                                          [10.34339361, 1.85744069, 1.41517779, 9.59343932, 2.2050838,
                                           7.67097848],
                                          [13.89958062, 2.77055121, 0.257797, 11.71776589, 0.99092663,
                                           10.44102166],
                                          [11.03042704, 0.98221761, 0.17153512, 10.007944, 1.01396213,
                                           10.06233002]], dtype=jnp.float64)}
    precomputed_normalization_stats = DataStats(
        inputs=Stats(mean=jnp.array([0.00055799, 0.0285231, -0.00933083, -0.09942926, 0.08258638, -0.00132751],
                                    dtype=jnp.float64),
                     std=jnp.array([0.28729099, 0.70630461, 0.70727364, 2.27893656, 4.6260925, 0.55621416],
                                   dtype=jnp.float64)),
        outputs=Stats(
            mean=jnp.array([-0.00929227, 0.01625088, -0.01068899, 0.02509439, -0.01438436],
                           dtype=jnp.float64),
            std=jnp.array([0.22896878, 0.31788133, 0.32660905, 1.29298522, 1.11480673],
                          dtype=jnp.float64)))
    precomputed_function_norms = jnp.array([14.77733678, 13.75797717, 13.80373648, 16.40662952, 17.46610356],
                                           dtype=jnp.float64)

    key = jr.PRNGKey(seed)
    key, key_offline_data = jr.split(key)

    offline_data_sampler = CartPoleOfflineData(action_repeat=action_repeat,
                                               predict_difference=True)

    offline_data = offline_data_sampler.sample(key=key_offline_data,
                                               num_samples=num_offline_data,
                                               max_abs_lin_position=1.0,
                                               max_abs_ang_velocity=5.0,
                                               max_abs_lin_velocity=5.0,
                                               )

    env = CartPoleEnv()

    if use_precomputed_kernel_params:
        num_training_steps = constant_schedule(0)
    else:
        num_training_steps = constant_schedule(num_training_steps)

    if use_function_norms:
        model = GPStatisticalModel(
            kernel=ARD(input_dim=env.observation_size + env.action_size),
            input_dim=env.observation_size + env.action_size,
            output_dim=env.observation_size,
            output_stds=1e-3 * jnp.ones(shape=(env.observation_size,)),
            logging_wandb=log_wandb,
            beta=None,
            f_norm_bound=precomputed_function_norms * beta,
            fixed_kernel_params=use_precomputed_kernel_params,
            normalization_stats=precomputed_normalization_stats,
            num_training_steps=num_training_steps,
        )
    else:
        model = GPStatisticalModel(
            kernel=ARD(input_dim=env.observation_size + env.action_size),
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
    class CartPoleRewardParams:
        control_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.01))
        angle_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
        pos_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(1.0))
        vel_cost: chex.Array = struct.field(default_factory=lambda: jnp.array(0.1))
        target_angle: chex.Array = struct.field(default_factory=lambda: jnp.array(jnp.pi))

    class CartPoleReward(Reward):
        def __init__(self, target_angle: float = jnp.pi):
            super().__init__(x_dim=5, u_dim=1)
            self.target_angle = jnp.array(target_angle)

        @staticmethod
        def cos_sin_to_angle_representation(cos_sin_angle: Float[Array, '2']) -> Scalar:
            return jnp.arctan2(cos_sin_angle[1], cos_sin_angle[0])

        def from_obs_to_state(self, state: Float[Array, '5']) -> Float[Array, '4']:
            assert state.shape == (5,)
            position, cos, sin, linear_velocity, angular_velocity = state[0], state[1], state[2], state[3], state[4]
            angle = self.cos_sin_to_angle_representation(jnp.array([cos, sin]))
            return jnp.array([position, angle, linear_velocity, angular_velocity])

        def __call__(self,
                     x: chex.Array,
                     u: chex.Array,
                     reward_params: CartPoleRewardParams,
                     x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
            chex.assert_shape(x, (self.x_dim,))
            chex.assert_shape(u, (self.u_dim,))
            chex.assert_shape(x_next, (self.x_dim,))
            # get intrinsic reward out
            x_compressed = self.from_obs_to_state(x)
            position, angle = x_compressed[0], x_compressed[1]
            linear_velocity, angular_velocity = x_compressed[2], x_compressed[3]

            target_angle = reward_params.target_angle
            diff_th = angle - target_angle
            diff_th = ((diff_th + jnp.pi) % (2 * jnp.pi)) - jnp.pi
            reward = -(reward_params.angle_cost * diff_th ** 2 + reward_params.pos_cost * position ** 2 +
                       reward_params.vel_cost * (
                               linear_velocity ** 2 + angular_velocity ** 2)) - reward_params.control_cost * u[
                         0] ** 2
            reward = reward.squeeze()
            return Normal(loc=reward, scale=jnp.zeros_like(reward)), reward_params

        def init_params(self, key: chex.PRNGKey) -> CartPoleRewardParams:
            default_reward_params = CartPoleRewardParams()
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

    cost_fn = PositionBoundBinary(horizon=icem_horizon,
                                  max_position=max_position,
                                  violation_eps=violation_eps, )

    agent = alg(
        env=CartPoleEnv(),
        model=model,
        episode_length=episode_length,
        action_repeat=action_repeat,
        cost_fn=cost_fn,
        test_tasks=[Task(reward=CartPoleReward(target_angle=0.0), name='Keep down', env=env),
                    Task(reward=CartPoleReward(target_angle=jnp.pi), name='Swing up', env=env),
                    ],
        predict_difference=True,
        num_training_steps=num_training_steps,
        icem_horizon=icem_horizon,
        icem_params=icem_params,
        log_to_wandb=log_wandb,
        use_pessimism=use_pessimism,
        use_optimism=use_optimism,
        optimizer=optimizer,
    )

    model_state = model.init(jr.PRNGKey(seed))

    # Here we set the model_state of the GP to the right one, we must ensure that we don't do
    # any GP training afterward
    if use_precomputed_kernel_params:
        model_state.model_state.params = precomputed_kernel_params
        model_state.model_state.data_stats = precomputed_normalization_stats

    # Here we need to take care of the first datapoint!!
    model_state.model_state.history = Data(inputs=jnp.array([[0., 1.0, 0., 0., 0., 0.]]),
                                           outputs=jnp.array([[0., 0., 0., 0., 0.]]))

    if log_wandb:
        import os
        if os.environ.get('WANDB_PROJECT') is not None:
            project_name = os.environ.get('WANDB_PROJECT')
        if os.environ.get('WANDB_ENTITY') is not None:
            entity_name = os.environ.get('WANDB_ENTITY')
        wandb.init(project=project_name,
                   config=configs,
                   entity=entity_name,
                   dir=logs_dir,
                   )
    env_name = 'cartpole'
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
    import os
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
        max_position=args.max_position,
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
        optimizer=args.optimizer,
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
    parser.add_argument('--num_steps', type=int, default=5)
    parser.add_argument('--exponent', type=float, default=1.0)
    parser.add_argument('--lambda_constraint', type=float, default=1e8)
    parser.add_argument('--icem_horizon', type=int, default=50)
    parser.add_argument('--episode_length', type=int, default=50)
    parser.add_argument('--action_repeat', type=int, default=2)
    parser.add_argument('--max_position', type=float, default=1.5)
    parser.add_argument('--num_training_steps', type=int, default=1_000)
    parser.add_argument('--use_optimism', type=int, default=1)
    parser.add_argument('--use_pessimism', type=int, default=1)
    parser.add_argument('--log_wandb', type=int, default=1)
    parser.add_argument('--num_gpus', type=int, default=0)
    parser.add_argument('--function_norm', type=float, default=1.0)
    parser.add_argument('--num_elites', type=int, default=50)
    parser.add_argument('--beta', type=float, default=2.0)
    parser.add_argument('--use_precomputed_kernel_params', type=int, default=0)
    parser.add_argument('--use_function_norms', type=int, default=0)
    parser.add_argument('--num_offline_data', type=int, default=50)
    parser.add_argument('--violation_eps', type=float, default=0.1)
    parser.add_argument('--optimizer', type=str, default='icem')

    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--exp_result_folder', type=str, default=None)

    args = parser.parse_args()
    main(args)
