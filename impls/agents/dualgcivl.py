import copy
from typing import Any

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    GCActor,
    GCDiscreteActor,
    GCValue,
    GCBilinearReprValue,
    Identity,
)


class DualGCIVLAgent(flax.struct.PyTreeNode):

    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def rep_loss(self, batch, grad_params):
        """IQL-style loss for training the dual goal representation modules."""

        # Rep value loss.
        q1, q2 = self.network.select('target_rep_critic')(
            batch['observations'], batch['value_goals'], batch['actions']
        )
        q = jnp.minimum(q1, q2)
        v = self.network.select('rep_value')(
            batch['observations'], batch['value_goals'], params=grad_params
        )
        value_loss = self.expectile_loss(q - v, q - v, self.config['rep_expectile']).mean()

        # Rep critic loss.
        next_v = self.network.select('rep_value')(batch['next_observations'], batch['value_goals'])
        td_q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1c, q2c = self.network.select('rep_critic')(
            batch['observations'], batch['value_goals'], batch['actions'], params=grad_params
        )
        critic_loss = ((q1c - td_q) ** 2 + (q2c - td_q) ** 2).mean()

        return value_loss + critic_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'critic_loss': critic_loss,
            'q_mean': td_q.mean(),
            'q_max': td_q.max(),
            'q_min': td_q.min(),
        }

    def _goal_reps(self, goals):
        """Return goal representation vector ψ(goals) from rep_value."""
        reps = self.network.select('rep_value')(goals)
        if self.config.get('main_stopgrad_rep', True):
            reps = jax.lax.stop_gradient(reps)
        return reps

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss, conditioning on dual goal reps."""
        goal_reps = self._goal_reps(batch['value_goals'])

        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], goal_reps)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], goal_reps)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], goal_reps, params=grad_params)
        v = (v1 + v2) / 2

        value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
        value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
        value_loss = value_loss1 + value_loss2

        return value_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        """Compute the AWR actor loss, conditioning on dual goal reps."""
        goal_reps = self._goal_reps(batch['actor_goals'])

        v1, v2 = self.network.select('value')(batch['observations'], goal_reps)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], goal_reps)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('actor')(batch['observations'], goal_reps, params=grad_params)
        log_prob = dist.log_prob(batch['actions'])

        actor_loss = -(exp_a * log_prob).mean()

        actor_info = {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
        }
        if not self.config['discrete']:
            actor_info.update(
                {
                    'mse': jnp.mean((dist.mode() - batch['actions']) ** 2),
                    'std': jnp.mean(dist.scale_diag),
                }
            )

        return actor_loss, actor_info


    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        rep_loss, rep_info = self.rep_loss(batch, grad_params)
        for k, v in rep_info.items():
            info[f'rep/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        for k, v in actor_info.items():
            info[f'actor/{k}'] = v

        loss = value_loss + actor_loss + rep_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with info."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'value')
        self.target_update(new_network, 'rep_critic')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations, goals=None, seed=None, temperature=1.0):
        """Sample actions from the actor."""
        goal_reps = self._goal_reps(goals)
        dist = self.network.select('actor')(observations, goal_reps, temperature=temperature)
        actions = dist.sample(seed=seed)
        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(cls, seed, ex_observations, ex_actions, config, ex_goals=None):
        """Create a new agent."""
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        # Raw goals for representation learning follow the original GC setting.
        if ex_goals is None:
            ex_goals = ex_observations

        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        rep_dim = config['dual_repr_dim']
        ex_goal_reps = jnp.zeros(shape=(1, rep_dim))

        # Encoders:
        # - rep_* take RAW goals.
        # - value/actor take pre-computed goal reps (goal_encoder=Identity()).
        if config.get('encoder') is not None:
            encoder_module = encoder_modules[config['encoder']]

            rep_value_state_enc = GCEncoder(state_encoder=encoder_module())
            rep_value_goal_enc = GCEncoder(state_encoder=encoder_module())
            rep_critic_enc = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            target_rep_critic_enc = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())

            value_enc = GCEncoder(state_encoder=encoder_module(), goal_encoder=Identity())
            target_value_enc = GCEncoder(state_encoder=encoder_module(), goal_encoder=Identity())
            actor_enc = GCEncoder(state_encoder=encoder_module(), goal_encoder=Identity())
        else:
            rep_value_state_enc = None
            rep_value_goal_enc = GCEncoder(state_encoder=Identity())
            rep_critic_enc = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())
            target_rep_critic_enc = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())

            value_enc = None
            target_value_enc = None
            actor_enc = None

        # Main value + actor.
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_enc,
        )

        if config['discrete']:
            actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=actor_enc,
            )
        else:
            actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=actor_enc,
            )

        # Dual representation modules (same architecture style as newdualhiql).
        rep_value_def = GCBilinearReprValue(
            hidden_dims=config['rep_hidden_dims'],
            latent_dim=rep_dim,
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=rep_value_state_enc,
            goal_encoder=rep_value_goal_enc,
            ret_mean=True,
        )

        rep_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=rep_critic_enc,
        )
        target_rep_critic_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_rep_critic_enc,
        )

        network_info = dict(
            rep_value=(rep_value_def, (ex_observations, ex_goals)),
            rep_critic=(rep_critic_def, (ex_observations, ex_goals, ex_actions)),
            target_rep_critic=(copy.deepcopy(target_rep_critic_def), (ex_observations, ex_goals, ex_actions)),
            value=(value_def, (ex_observations, ex_goal_reps)),
            target_value=(copy.deepcopy(value_def), (ex_observations, ex_goal_reps)),
            actor=(actor_def, (ex_observations, ex_goal_reps)),
        )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_rep_critic'] = params['modules_rep_critic']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='dualgcivl',
            lr=3e-4,
            batch_size=1024,
            rep_hidden_dims=(512, 512, 512),
            actor_hidden_dims=(512, 512, 512),
            value_hidden_dims=(512, 512, 512),
            layer_norm=True,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            alpha=10.0,
            const_std=True,
            discrete=False,
            # Representation-learning hyperparameters.
            rep_expectile=0.7,
            dual_repr_dim=256,
            main_stopgrad_rep=True,
            # Dataset hyperparameters.
            dataset_class='GCDataset',
            oraclerep=False,
            norm=False,
            value_p_curgoal=0.2,
            value_p_trajgoal=0.5,
            value_p_randomgoal=0.3,
            value_geom_sample=True,
            actor_p_curgoal=0.0,
            actor_p_trajgoal=1.0,
            actor_p_randomgoal=0.0,
            actor_geom_sample=False,
            gc_negative=True,
            p_aug=0.0,
            frame_stack=ml_collections.config_dict.placeholder(int),
            encoder=None,
        )
    )
    return config
