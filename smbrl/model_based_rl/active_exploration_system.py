from typing import Generic, Tuple

import chex
import jax.numpy as jnp
import jax.random
import jax.random as jr
from bsm.statistical_model import StatisticalModel
from bsm.utils.type_aliases import ModelState
from distrax import Distribution, Normal
from jaxtyping import Float, Array, Scalar
from mbpo.systems.base_systems import SystemParams, SystemState, System
from mbpo.systems.dynamics.base_dynamics import Dynamics
from mbpo.systems.dynamics.base_dynamics import DynamicsParams as DummyDynamicsParams
from mbpo.systems.rewards.base_rewards import Reward, RewardParams


@chex.dataclass
class DynamicsParams(Generic[ModelState, DummyDynamicsParams]):
    key: chex.PRNGKey
    model_state: ModelState


class ExplorationDynamics(Dynamics, Generic[ModelState]):
    def __init__(self,
                 x_dim: int,
                 u_dim: int,
                 model: StatisticalModel,
                 use_log: bool = True,
                 scale_with_aleatoric_std: bool = True,
                 aleatoric_noise_in_prediction: bool = True,
                 predict_difference: bool = True,
                 ):
        Dynamics.__init__(self, x_dim=x_dim, u_dim=u_dim)
        self.model = model
        self.use_log = use_log
        self.scale_with_aleatoric_std = scale_with_aleatoric_std
        self.aleatoric_noise_in_prediction = aleatoric_noise_in_prediction
        self.predict_difference = predict_difference

    def init_params(self, key: chex.PRNGKey) -> DynamicsParams:
        param_key, model_state_key = jr.split(key, 2)
        model_state = self.model.init(model_state_key)
        return DynamicsParams(key=key, model_state=model_state)

    def get_intrinsic_reward(self,
                             epistemic_std: Float[Array, '... observation_dim'],
                             aleatoric_std: Float[Array, '... observation_dim']) -> Scalar:
        if self.scale_with_aleatoric_std:
            # sigma^2_ep / sigma^2_al
            intrinsic_reward = jnp.square(epistemic_std / jnp.clip(aleatoric_std, min=1e-4))
        else:
            # sigma^2_ep
            intrinsic_reward = jnp.square(epistemic_std)
        if self.use_log:
            # use log transform
            intrinsic_reward = jnp.log(1 + intrinsic_reward)
        # sum over the state axis
        return jnp.sum(intrinsic_reward, axis=0)

    def next_state(self,
                   x: chex.Array,
                   u: chex.Array,
                   dynamics_params: DynamicsParams) -> Tuple[Distribution, DynamicsParams]:
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        # Create state-action pair
        z = jnp.concatenate([x, u])
        next_key, key_sample_x_next = jr.split(dynamics_params.key)
        pred = self.model(z, dynamics_params.model_state)
        epistemic_std, aleatoric_std = pred.epistemic_std, pred.aleatoric_std
        beta = pred.statistical_model_state.beta
        x_next = x
        if self.predict_difference:
            x_next += pred.mean + beta * epistemic_std * jr.normal(key=key_sample_x_next, shape=pred.mean.shape)
        else:
            x_next = pred.mean + beta * epistemic_std * jr.normal(key=key_sample_x_next, shape=pred.mean.shape)

        intrinsic_reward = self.get_intrinsic_reward(epistemic_std, aleatoric_std)
        intrinsic_reward = jnp.atleast_1d(intrinsic_reward)

        if not self.aleatoric_noise_in_prediction:
            aleatoric_std = 0 * aleatoric_std
        # add intrinsic reward to the next state
        x_next_with_reward = jnp.concatenate([x_next, intrinsic_reward], axis=-1)
        aleatoric_std_with_reward = jnp.concatenate([aleatoric_std, jnp.zeros_like(intrinsic_reward)], axis=-1)
        new_dynamics_params = dynamics_params.replace(key=next_key)
        return Normal(loc=x_next_with_reward, scale=aleatoric_std_with_reward), new_dynamics_params


@chex.dataclass
class ExplorationRewardParams:
    action_cost: chex.Array | float = 0.0


class ExplorationReward(Reward, ExplorationRewardParams):
    def __init__(self, x_dim: int, u_dim: int):
        super().__init__(x_dim=x_dim, u_dim=u_dim)

    def __call__(self,
                 x: chex.Array,
                 u: chex.Array,
                 reward_params: ExplorationRewardParams,
                 x_next: chex.Array | None = None) -> Tuple[Distribution, RewardParams]:
        chex.assert_shape(x, (self.x_dim,))
        chex.assert_shape(u, (self.u_dim,))
        chex.assert_shape(x_next, (self.x_dim + 1,))
        # get intrinsic reward out
        intrinsic_reward = x_next[-1]
        total_reward = intrinsic_reward - reward_params.action_cost * jnp.sum(jnp.square(u), axis=0)
        return Normal(loc=total_reward, scale=jnp.zeros_like(total_reward)), reward_params

    def init_params(self, key: chex.PRNGKey) -> ExplorationRewardParams:
        return ExplorationRewardParams()


class ExplorationSystem(System, Generic[ModelState, RewardParams]):
    def __init__(self, dynamics: ExplorationDynamics[ModelState], reward: Reward[RewardParams] | None = None):
        if reward is None:
            reward = ExplorationReward(x_dim=dynamics.x_dim, u_dim=dynamics.u_dim)
        super().__init__(dynamics, reward)
        self.dynamics = dynamics
        self.reward = reward
        self.x_dim = dynamics.x_dim
        self.u_dim = dynamics.u_dim

    def get_reward(self,
                   x: chex.Array,
                   u: chex.Array,
                   reward_params: RewardParams,
                   x_next: chex.Array,
                   key: jax.random.PRNGKey):
        # x_next includes the next state and the intrinsic reward
        chex.assert_shape(x_next, (self.x_dim + 1,))
        if isinstance(self.reward, ExplorationReward):
            # include the intrinsic reward in x_next
            reward_dist, new_reward_params = self.reward(x, u, reward_params, x_next)
        else:
            # ignore the last state in x_next which is the intrinsic reward
            reward_dist, new_reward_params = self.reward(x, u, reward_params, x_next[:-1])
        reward = reward_dist.sample(seed=key)
        return reward, new_reward_params

    def step(self,
             x: chex.Array,
             u: chex.Array,
             system_params: SystemParams[ModelState, RewardParams],
             ) -> SystemState:
        """

        :param x: current state of the system
        :param u: current action of the system
        :param system_params: parameters of the system
        :return: Tuple of next state, reward, updated system parameters
        """
        assert x.shape == (self.x_dim,) and u.shape == (self.u_dim,)
        x_next_dist, new_dynamics_params = self.dynamics.next_state(x, u, system_params.dynamics_params)
        next_state_key, reward_key, new_systems_key = jr.split(system_params.key, 3)
        x_next = x_next_dist.sample(seed=next_state_key)
        reward, new_reward_params = self.get_reward(x, u, system_params.reward_params, x_next, reward_key)
        new_systems_params = system_params.replace(dynamics_params=new_dynamics_params,
                                                   reward_params=new_reward_params,
                                                   key=new_systems_key)
        new_system_state = SystemState(
            x_next=x_next[:-1],
            reward=reward,
            system_params=new_systems_params,
            done=jnp.array(0.0),
        )
        return new_system_state
