import copy
from typing import Any, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import ml_collections
import optax
from utils.encoders import GCEncoder, encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import (
    MLP,
    GCActor,
    GCDiscreteActor,
    GCValue,
    Identity,
    LengthNormalize,
    GCBilinearReprValue,
)


class ProjMLP(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int
    layer_norm: bool = True
    use_state: bool = False
    obs_encoder: nn.Module = None

    def setup(self):
        self.mlp = MLP(
            hidden_dims=(*self.hidden_dims, self.output_dim),
            activate_final=False,
            layer_norm=self.layer_norm,
        )
    
    @staticmethod
    def safe_l2_norm(x, axis=-1, keepdims=True, eps=1e-12):
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
        m = jnp.maximum(m, eps)
        y = x / m
        s = jnp.sum(y * y, axis=axis, keepdims=keepdims)
        n = m * jnp.sqrt(s + eps)
        return n.astype(x.dtype)

    def safe_length_normalize(self, x, axis=-1, eps=1e-12):
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n = self.safe_l2_norm(x, axis=axis, keepdims=True, eps=eps)
        return x / n

    def __call__(self, observations, dual_inputs):
        x = dual_inputs
        if self.use_state:
            if self.obs_encoder is not None:
                s = self.obs_encoder(observations)
            else:
                s = observations
            x = jnp.concatenate([s, dual_inputs], axis=-1)
        x = self.mlp(x)
        x = self.safe_length_normalize(x, axis=-1) * jnp.sqrt(x.shape[-1])
        return x


class DualHIQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        """Compute the expectile loss."""
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)
    
    @staticmethod
    def safe_l2_norm(x, axis=-1, keepdims=True, eps=1e-12):
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        m = jnp.max(jnp.abs(x), axis=axis, keepdims=True)
        m = jnp.maximum(m, eps)
        y = x / m
        s = jnp.sum(y * y, axis=axis, keepdims=keepdims)
        n = m * jnp.sqrt(s + eps)
        return n.astype(x.dtype)

    def safe_length_normalize(self, x, axis=-1, eps=1e-12):
        x = jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        n = self.safe_l2_norm(x, axis=axis, keepdims=True, eps=eps)
        return x / n

    def _dual_goal(self, goals, goal_encoded=False, grad_params=None):
        return self.network.select('dual_repr_value')(goals, goal_encoded=goal_encoded, params=grad_params)

    def _dual_analogy(self, observations, goals, obs_encoded=False, goal_encoded=False, grad_params=None):
        """Get the dual analogy representation \psi[goals] - \psi[observations]."""
        obs_repr = self._dual_goal(observations, goal_encoded=obs_encoded, grad_params=grad_params)
        goal_repr = self._dual_goal(goals, goal_encoded=goal_encoded, grad_params=grad_params)
        dual_analogies = goal_repr - obs_repr
        return dual_analogies

    def _stopgrad_dual_goal(self, x, goal_encoded=False):
        z = self._dual_goal(x, goal_encoded=goal_encoded, grad_params=None)
        return jax.lax.stop_gradient(z)

    def _dual_input(self, observations, goals, use_analogy=False):
        """Dual input used by goal_repr. Always stop-grad w.r.t. dual representation parameters."""
        if use_analogy:
            return self._stopgrad_dual_goal(goals) - self._stopgrad_dual_goal(observations)
        return self._stopgrad_dual_goal(goals)

    def _goal_repr(self, observations, goals, allow_grad, grad_params):
        """Compute phi(s, g) using dual embeddings in place of raw goals.

        - allow_grad=True  -> params=grad_params (goal_repr can be trained)
        - allow_grad=False -> params from self.network (goal_repr not updated by this loss)
        """
        dual_inp = self._dual_input(observations, goals, use_analogy=self.config['use_analogy'])
        if allow_grad:
            return self.network.select('goal_repr')(observations, dual_inp, params=grad_params)
        return self.network.select('goal_repr')(observations, dual_inp)

    def dual_repr_loss(self, batch, grad_params):
        """Dual representation loss."""
        q1, q2 = self.network.select('target_dual_repr_critic')(batch['observations'], batch['value_goals'], batch['actions'])
        q = jnp.minimum(q1, q2)
        v = self.network.select('dual_repr_value')(batch['observations'], batch['value_goals'], params=grad_params)
        value_loss = self.expectile_loss(q - v, q - v, self.config['dual_repr_expectile']).mean()

        next_v = self.network.select('dual_repr_value')(batch['next_observations'], batch['value_goals'])
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v
        q1, q2 = self.network.select('dual_repr_critic')(batch['observations'], batch['value_goals'], batch['actions'], params=grad_params)
        critic_loss = ((q1 - q) ** 2 + (q2 - q) ** 2).mean()

        return value_loss + critic_loss, {
            'value_loss': value_loss,
            'v_mean': v.mean(),
            'v_max': v.max(),
            'v_min': v.min(),
            'critic_loss': critic_loss,
            'q_mean': q.mean(),
            'q_max': q.max(),
            'q_min': q.min(),
        }

    def value_loss(self, batch, grad_params):
        """Compute the IVL value loss (HIQL-style), replacing g with dual-goal representations."""
        # Target-side goal repr (no grad to goal_repr / dual_repr).
        next_g = self._goal_repr(batch['next_observations'], batch['value_goals'], allow_grad=False, grad_params=grad_params)
        g_t = self._goal_repr(batch['observations'], batch['value_goals'], allow_grad=False, grad_params=grad_params)

        (next_v1_t, next_v2_t) = self.network.select('target_value')(batch['next_observations'], next_g)
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        (v1_t, v2_t) = self.network.select('target_value')(batch['observations'], g_t)
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        # Current-side goal repr (value trains goal_repr).
        g = self._goal_repr(
            batch['observations'],
            batch['value_goals'],
            allow_grad=not self.config['value_rep_grad_stop'],
            grad_params=grad_params,
        )

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(batch['observations'], g, params=grad_params)
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

    def low_actor_loss(self, batch, grad_params):
        """Compute the low-level actor loss (HIQL-style).

        - value always uses frozen goal_repr (for adv), so policy weights don't train goal_repr.
        - low-level actor may (optionally) train goal_repr, controlled by policy_rep_grad_stop.
        """
        # Frozen goal repr for advantage computation.
        g_v = self._goal_repr(batch['observations'], batch['low_actor_goals'], allow_grad=False, grad_params=grad_params)
        ng_v = self._goal_repr(batch['next_observations'], batch['low_actor_goals'], allow_grad=False, grad_params=grad_params)

        v1, v2 = self.network.select('value')(batch['observations'], g_v)
        nv1, nv2 = self.network.select('value')(batch['next_observations'], ng_v)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v
        adv = jax.lax.stop_gradient(adv)

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Trainable (optional) goal repr for low-level policy.
        g_pi = self._goal_repr(
            batch['observations'],
            batch['low_actor_goals'],
            allow_grad=True,
            grad_params=grad_params,
        )
        if self.config['policy_rep_grad_stop']:
            g_pi = jax.lax.stop_gradient(g_pi)

        dist = self.network.select('low_actor')(batch['observations'], g_pi, params=grad_params)
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

    def high_actor_loss(self, batch, grad_params):
        """Compute the high-level actor loss (HIQL-style).

        High-level loss should NOT update goal_repr; both conditioning and targets use frozen goal_repr.
        """
        # Frozen goal repr for advantage computation.
        g_v = self._goal_repr(batch['observations'], batch['high_actor_goals'], allow_grad=False, grad_params=grad_params)
        ng_v = self._goal_repr(batch['high_actor_targets'], batch['high_actor_goals'], allow_grad=False, grad_params=grad_params)

        v1, v2 = self.network.select('value')(batch['observations'], g_v)
        nv1, nv2 = self.network.select('value')(batch['high_actor_targets'], ng_v)
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        # Frozen goal repr for conditioning.
        dist = self.network.select('high_actor')(batch['observations'], g_v, params=grad_params)

        # Frozen target representation in goal_repr space.
        target = self._goal_repr(batch['observations'], batch['high_actor_targets'], allow_grad=False, grad_params=grad_params)
        target = jax.lax.stop_gradient(target)

        log_prob = dist.log_prob(target)
        actor_loss = -(exp_a * log_prob).mean()

        return actor_loss, {
            'actor_loss': actor_loss,
            'adv': adv.mean(),
            'bc_log_prob': log_prob.mean(),
            'mse': jnp.mean((dist.mode() - target) ** 2),
            'std': jnp.mean(dist.scale_diag),
        }

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        dual_repr_loss, dual_repr_info = self.dual_repr_loss(batch, grad_params)
        for k, v in dual_repr_info.items():
            info[f'dual_repr/{k}'] = v

        value_loss, value_info = self.value_loss(batch, grad_params)
        for k, v in value_info.items():
            info[f'value/{k}'] = v

        low_actor_loss, low_actor_info = self.low_actor_loss(batch, grad_params)
        for k, v in low_actor_info.items():
            info[f'low_actor/{k}'] = v

        high_actor_loss, high_actor_info = self.high_actor_loss(batch, grad_params)
        for k, v in high_actor_info.items():
            info[f'high_actor/{k}'] = v

        loss = dual_repr_loss + value_loss + low_actor_loss + high_actor_loss
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
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'dual_repr_critic')
        self.target_update(new_network, 'value')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(
        self,
        observations,
        goals=None,
        seed=None,
        temperature=1.0,
    ):
        """Sample actions from the actor.

        It first queries the high-level actor to obtain subgoal representations, and then queries the low-level actor
        to obtain raw actions.
        """
        high_seed, low_seed = jax.random.split(seed)

        g_pi = self._goal_repr(observations, goals, allow_grad=False, grad_params=None)
        high_dist = self.network.select('high_actor')(observations, g_pi, temperature=temperature)
        goal_reprs = high_dist.sample(seed=high_seed)
        goal_reprs = self.safe_length_normalize(goal_reprs, axis=-1) * jnp.sqrt(goal_reprs.shape[-1])

        low_dist = self.network.select('low_actor')(observations, goal_reprs, temperature=temperature)
        actions = low_dist.sample(seed=low_seed)

        if not self.config['discrete']:
            actions = jnp.clip(actions, -1, 1)
        return actions

    @classmethod
    def create(
        cls,
        seed,
        ex_observations,
        ex_actions,
        config,
    ):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example batch of observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_goals = ex_observations
        if config['discrete']:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]
        ex_dual = jnp.zeros((ex_observations.shape[0], config['dual_repr_dim']), dtype=jnp.float32)
        ex_goal_repr = jnp.zeros((ex_observations.shape[0], config['goal_repr_dim']), dtype=jnp.float32)

        # Define (optionally state-dependent) subgoal representation phi(psi(g)) or phi([s; psi(g)]).
        use_state_in_goal_repr = (config['subgoal_repr_type'] == 'dual_state')
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            obs_encoder_def = encoder_module() if use_state_in_goal_repr else None
        else:
            encoder_module = None
            obs_encoder_def = None

        goal_repr_def = ProjMLP(
            hidden_dims=config['value_hidden_dims'],
            output_dim=config['goal_repr_dim'],
            layer_norm=config['layer_norm'],
            use_state=use_state_in_goal_repr,
            obs_encoder=obs_encoder_def,
        )

        # Encoders:
        # - dual_repr_* take RAW states/goals (not goal_repr).
        # - value/actors take pre-computed goal_repr vectors (goal_encoder=Identity()).
        if config['encoder'] is not None:
            # Dual repr (raw pixel goals)
            dual_repr_value_state_encoder_def = GCEncoder(state_encoder=encoder_module())
            dual_repr_value_goal_encoder_def = GCEncoder(state_encoder=encoder_module())
            dual_repr_critic_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            target_dual_repr_critic_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            # Value / actors (goal_repr vectors)
            value_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=Identity())
            target_value_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=Identity())
            low_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=Identity())
            high_actor_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=Identity())
        else:
            # Dual repr (raw vector goals)
            dual_repr_value_state_encoder_def = None
            dual_repr_value_goal_encoder_def = GCEncoder(state_encoder=Identity())
            dual_repr_critic_encoder_def = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())
            target_dual_repr_critic_encoder_def = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())
            # Value / actors (goal_repr vectors)
            value_encoder_def = None
            target_value_encoder_def = None
            low_actor_encoder_def = None
            high_actor_encoder_def = None

        # dual representation
        dual_repr_value_def = GCBilinearReprValue(
            hidden_dims=config['dual_repr_hidden_dims'],
            latent_dim=config['dual_repr_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            value_exp=False,
            state_encoder=dual_repr_value_state_encoder_def,
            goal_encoder=dual_repr_value_goal_encoder_def,
            ret_mean=True,
        )

        dual_repr_critic_def = GCValue(
            hidden_dims=config['dual_repr_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=dual_repr_critic_encoder_def,
        )
        target_dual_repr_critic_def = GCValue(
            hidden_dims=config['dual_repr_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_dual_repr_critic_encoder_def,
        )

        # Define value and actor networks (condition on goal_repr vectors).
        value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=value_encoder_def,
        )
        target_value_def = GCValue(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            gc_encoder=target_value_encoder_def,
        )

        if config['discrete']:
            low_actor_def = GCDiscreteActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                gc_encoder=low_actor_encoder_def,
            )
        else:
            low_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=low_actor_encoder_def,
            )

        high_actor_def = GCActor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=config['goal_repr_dim'],
            state_dependent_std=False,
            const_std=config['const_std'],
            gc_encoder=high_actor_encoder_def,
        )

        network_info = dict(
            goal_repr=(goal_repr_def, (ex_observations, ex_dual)),
            dual_repr_value=(dual_repr_value_def, (ex_observations, ex_goals)),
            dual_repr_critic=(dual_repr_critic_def, (ex_observations, ex_goals, ex_actions)),
            target_dual_repr_critic=(target_dual_repr_critic_def, (ex_observations, ex_goals, ex_actions)),
            value=(value_def, (ex_observations, ex_goal_repr)),
            target_value=(target_value_def, (ex_observations, ex_goal_repr)),
            low_actor=(low_actor_def, (ex_observations, ex_goal_repr)),
            high_actor=(high_actor_def, (ex_observations, ex_goal_repr)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config['lr'])
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_dual_repr_critic'] = params['modules_dual_repr_critic']
        params['modules_target_value'] = params['modules_value']

        return cls(rng, network=network, config=flax.core.FrozenDict(**config))


def get_config():
    config = ml_collections.ConfigDict(
        dict(
            # Agent hyperparameters.
            agent_name='dualhiql',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            goal_repr_hidden_dims=(256, 256),  # Subgoal representation hidden dimensions.
            dual_repr_hidden_dims=(512, 512, 512),  # Dual representation hidden dimensions.
            actor_hidden_dims=(512, 512, 512),  # Actor network hidden dimensions.
            value_hidden_dims=(512, 512, 512),  # Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            dual_repr_expectile=0.7,  # Dual representation expectile.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=20,  # Subgoal steps.
            goal_repr_dim=32,  # Goal representation dimension.
            dual_repr_dim=256,  # Dual representation latent dimension.
            use_analogy=False,  # Whether to condition goal_repr on analogy psi(g)-psi(s) instead of psi(g).
            subgoal_repr_type='dual',  # 'dual' or 'dual_state'
            const_std=True,  # Whether to use constant standard deviation for the actors.
            discrete=False,  # Whether the action space is discrete.
            encoder=ml_collections.config_dict.placeholder(str),  # Visual encoder name (None, 'impala_small', etc.).
            # Dataset hyperparameters.
            dataset_class='HGCDataset',  # Dataset class name.
            value_p_curgoal=0.2,  # Probability of using the current state as the value goal.
            value_p_trajgoal=0.5,  # Probability of using a future state in the same trajectory as the value goal.
            value_p_randomgoal=0.3,  # Probability of using a random state as the value goal.
            value_geom_sample=True,  # Whether to use geometric sampling for future value goals.
            actor_p_curgoal=0.0,  # Probability of using the current state as the actor goal.
            actor_p_trajgoal=1.0,  # Probability of using a future state in the same trajectory as the actor goal.
            actor_p_randomgoal=0.0,  # Probability of using a random state as the actor goal.
            actor_geom_sample=False,  # Whether to use geometric sampling for future actor goals.
            gc_negative=True,  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' (False) as reward.
            p_aug=0.0,  # Probability of applying image augmentation.
            frame_stack=ml_collections.config_dict.placeholder(int),  # Number of frames to stack.
            value_rep_grad_stop=False,  # Whether to stop gradients from value loss to goal_repr
            policy_rep_grad_stop=True,  # Whether to stop gradients from low-actor loss to goal_repr
        )
    )
    return config
