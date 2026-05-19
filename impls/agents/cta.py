
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
    ProjectedGCBilinearValue,
    ProjectedGCBilinearActor,
)


class ProjMLP(nn.Module):
    """Subgoal representation module built on top of *dual analogies*
    """
    hidden_dims: Sequence[int]
    repr_dim: int
    layer_norm: bool = True
    concat_state: bool = False
    state_encoder: nn.Module = None

    def setup(self):
        self._mlp = MLP(
            hidden_dims=(*self.hidden_dims, self.repr_dim),
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

    def __call__(self, dual_analogy, observations=None):
        if self.concat_state:
            assert observations is not None, "observations must be provided when concat_state=True."
            if self.state_encoder is not None:
                state_feat = self.state_encoder(observations)
            else:
                state_feat = observations
            x = jnp.concatenate([dual_analogy, state_feat], axis=-1)
        else:
            x = dual_analogy

        x = self._mlp(x)
        x = self.safe_length_normalize(x, axis=-1) * jnp.sqrt(x.shape[-1])
        return x


class CTAAgent(flax.struct.PyTreeNode):
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

    def _goal_repr(self, observations, goals, goal_encoded=False, grad_params=None):
        """Get the (state-dependent) subgoal representation z(s, g) in R^{goal_repr_dim}.
        """
        if goal_encoded:
            return goals

        dual_repr = self._dual_analogy(observations, goals, obs_encoded=False, goal_encoded=False, grad_params=None)
        dual_repr = jax.lax.stop_gradient(dual_repr)

        return self.network.select('goal_repr')(dual_repr, params=grad_params)

    def dual_repr_loss(self, batch, grad_params):
        """Dual representation loss.

        IMPORTANT: This is the only loss that updates dual_repr modules.
        """
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
        """Compute the IVL value loss.

        This value loss is similar to the original IQL value loss, but involves additional tricks to stabilize training.
        For example, when computing the expectile loss, we separate the advantage part (which is used to compute the
        weight) and the difference part (which is used to compute the loss), where we use the target value function to
        compute the former and the current value function to compute the latter. This is similar to how double DQN
        mitigates overestimation bias.
        """
        next_z = self._goal_repr(batch['next_observations'], batch['value_goals'])
        (next_v1_t, next_v2_t) = self.network.select('target_value')(next_z, batch['next_observations'])
        next_v_t = jnp.minimum(next_v1_t, next_v2_t)
        q = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v_t

        z_t = self._goal_repr(
            batch['observations'],
            batch['value_goals'],
            grad_params=grad_params if not self.config['value_rep_grad_stop'] else None,
        )
        (v1_t, v2_t) = self.network.select('target_value')(z_t, batch['observations'])
        v_t = (v1_t + v2_t) / 2
        adv = q - v_t

        q1 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v1_t
        q2 = batch['rewards'] + self.config['discount'] * batch['masks'] * next_v2_t
        (v1, v2) = self.network.select('value')(z_t, batch['observations'], params=grad_params)
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
        """Compute the low-level actor loss."""
        z = self._goal_repr(
            batch['observations'],
            batch['low_actor_goals'],
            grad_params=grad_params,
        )
        if self.config.get('policy_rep_grad_stop', False):
            z = jax.lax.stop_gradient(z)

        nz = self._goal_repr(batch['next_observations'], batch['low_actor_goals'])
        v1, v2 = self.network.select('value')(z, batch['observations'])
        nv1, nv2 = self.network.select('value')(nz, batch['next_observations'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['low_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        dist = self.network.select('low_actor')(z, batch['observations'], params=grad_params)
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
        """Compute the high-level actor loss."""
        z = self._goal_repr(batch['observations'], batch['high_actor_goals'])
        nz = self._goal_repr(batch['high_actor_targets'], batch['high_actor_goals'])
        v1, v2 = self.network.select('value')(z, batch['observations'])
        nv1, nv2 = self.network.select('value')(nz, batch['high_actor_targets'])
        v = (v1 + v2) / 2
        nv = (nv1 + nv2) / 2
        adv = nv - v

        exp_a = jnp.exp(adv * self.config['high_alpha'])
        exp_a = jnp.minimum(exp_a, 100.0)

        if self.config['use_bilinear_high_actor']:
            dist = self.network.select('high_actor')(z, batch['observations'], params=grad_params)
        else:
            dist = self.network.select('high_actor')(batch['observations'], batch['high_actor_goals'], params=grad_params)

        target = self._goal_repr(batch['observations'], batch['high_actor_targets'])
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

        if self.config['use_bilinear_high_actor']:
            z = self._goal_repr(observations, goals)   # (B, goal_repr_dim=32)
            high_dist = self.network.select('high_actor')(z, observations, temperature=temperature)
        else:
            high_dist = self.network.select('high_actor')(observations, goals, temperature=temperature)

        goal_reprs = high_dist.sample(seed=high_seed)
        goal_reprs = self.safe_length_normalize(goal_reprs, axis=-1) * jnp.sqrt(goal_reprs.shape[-1])

        low_dist = self.network.select('low_actor')(goal_reprs, observations, temperature=temperature)
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
        ex_analogies = jnp.zeros((ex_observations.shape[0], config['goal_repr_dim']), dtype=jnp.float32)
        ex_dual_reprs = jnp.zeros((ex_observations.shape[0], config['dual_repr_dim']), dtype=jnp.float32)

        # Define subgoal representation z(x) on top of the (frozen) dual representation psi(x).
        #   - dual      : z(x) = f(psi(x))
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]

        goal_repr_def = ProjMLP(
            hidden_dims=config['goal_repr_hidden_dims'],
            repr_dim=config['goal_repr_dim'],
            layer_norm=config['layer_norm'],
            concat_state=False,
            state_encoder=None,
        )

        # Define the encoders that handle the inputs to the dual representation networks.
        if config['encoder'] is not None:
            # Pixel-based environments require visual encoders for state inputs.

            # Dual repr
            dual_repr_value_state_encoder_def = GCEncoder(state_encoder=encoder_module())
            dual_repr_value_goal_encoder_def = GCEncoder(state_encoder=encoder_module())
            dual_repr_critic_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            target_dual_repr_critic_encoder_def = GCEncoder(state_encoder=encoder_module(), goal_encoder=encoder_module())
            # Value
            value_state_encoder_def = GCEncoder(state_encoder=encoder_module())
            value_analogy_encoder_def = None
            target_value_state_encoder_def = GCEncoder(state_encoder=encoder_module())
            target_value_analogy_encoder_def = None
            # Low-level actor
            low_actor_state_encoder_def = GCEncoder(state_encoder=encoder_module())
            low_actor_analogy_encoder_def = None
            # High-level actor
            if config['use_bilinear_high_actor']:
                high_actor_state_encoder_def = GCEncoder(state_encoder=encoder_module())
                high_actor_analogy_encoder_def = None
            else:
                high_actor_encoder_def = GCEncoder(concat_encoder=encoder_module())

        else:
            # State-based environments.
            # Dual repr
            dual_repr_value_state_encoder_def = None
            dual_repr_value_goal_encoder_def = GCEncoder(state_encoder=Identity())
            dual_repr_critic_encoder_def = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())
            target_dual_repr_critic_encoder_def = GCEncoder(state_encoder=Identity(), goal_encoder=Identity())
            # Value
            value_state_encoder_def = None
            value_analogy_encoder_def = None
            target_value_state_encoder_def = None
            target_value_analogy_encoder_def = None
            # Low-level actor
            low_actor_state_encoder_def = None
            low_actor_analogy_encoder_def = None
            # High-level actor
            if config['use_bilinear_high_actor']:
                high_actor_state_encoder_def = None
                high_actor_analogy_encoder_def = None
            else:
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

        # Define value and actor networks (on subgoal-space analogies).
        value_def = ProjectedGCBilinearValue(
            latent_dim=config['tr_latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            tr_anchor_hidden_dims=config['tr_anchor_hidden_dims'],
            tr_delta_hidden_dims=config['tr_delta_hidden_dims'],
            backbone_hidden_dims=config['value_hidden_dims'] if config['use_backbone'] else (),
            anchor_encoder=value_state_encoder_def,
            delta_encoder=value_analogy_encoder_def,
        )
        target_value_def = ProjectedGCBilinearValue(
            latent_dim=config['tr_latent_dim'],
            layer_norm=config['layer_norm'],
            ensemble=True,
            tr_anchor_hidden_dims=config['tr_anchor_hidden_dims'],
            tr_delta_hidden_dims=config['tr_delta_hidden_dims'],
            backbone_hidden_dims=config['value_hidden_dims'] if config['use_backbone'] else (),
            anchor_encoder=target_value_state_encoder_def,
            delta_encoder=target_value_analogy_encoder_def,
        )

        if config['discrete']:
            # TODO: implement low-level actor
            pass
        else:
            low_actor_def = ProjectedGCBilinearActor(
                latent_dim=config['tr_latent_dim'],
                layer_norm=config['layer_norm'],
                tr_anchor_hidden_dims=config['tr_anchor_hidden_dims'],
                tr_delta_hidden_dims=config['tr_delta_hidden_dims'],
                backbone_hidden_dims=config['actor_hidden_dims'] if config['use_backbone'] else (),
                action_dim=action_dim,
                state_dependent_std=False,
                const_std=config['const_std'],
                anchor_encoder=low_actor_state_encoder_def,
                delta_encoder=low_actor_analogy_encoder_def,
            )

        if config['use_bilinear_high_actor']:
            high_actor_def = ProjectedGCBilinearActor(
                latent_dim=config['tr_latent_dim'],
                layer_norm=config['layer_norm'],
                tr_anchor_hidden_dims=config['tr_anchor_hidden_dims'],
                tr_delta_hidden_dims=config['tr_delta_hidden_dims'],
                backbone_hidden_dims=config['actor_hidden_dims'] if config['use_backbone'] else (),
                action_dim=config['goal_repr_dim'],
                state_dependent_std=False,
                const_std=config['const_std'],
                anchor_encoder=high_actor_state_encoder_def,
                delta_encoder=high_actor_analogy_encoder_def,
            )
        else:
            high_actor_def = GCActor(
                hidden_dims=config['actor_hidden_dims'],
                action_dim=config['goal_repr_dim'],
                state_dependent_std=False,
                const_std=config['const_std'],
                gc_encoder=high_actor_encoder_def,
            )

        goal_repr_args = (ex_dual_reprs,)

        network_info = dict(
            goal_repr=(goal_repr_def, goal_repr_args),
            dual_repr_value=(dual_repr_value_def, (ex_observations, ex_goals)),
            dual_repr_critic=(dual_repr_critic_def, (ex_observations, ex_goals, ex_actions)),
            target_dual_repr_critic=(target_dual_repr_critic_def, (ex_observations, ex_goals, ex_actions)),
            value=(value_def, (ex_analogies, ex_observations)),
            target_value=(target_value_def, (ex_analogies, ex_observations)),
            low_actor=(low_actor_def, (ex_analogies, ex_observations)),
            high_actor=(high_actor_def, (ex_observations, ex_goals) if not config['use_bilinear_high_actor'] else (ex_analogies, ex_observations)),
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
            agent_name='cta',  # Agent name.
            lr=3e-4,  # Learning rate.
            batch_size=256,  # Batch size.
            goal_repr_hidden_dims=(256, 256),  # Subgoal representation hidden dimensions.
            dual_repr_hidden_dims=(512, 512, 512),  # Dual representation hidden dimensions.
            tr_anchor_hidden_dims=(128, 128, 128),  # Transductive anchor hidden dimensions.
            tr_delta_hidden_dims=(128, 128, 128),  # Transductive delta hidden dimensions.
            tr_latent_dim=8,  # Transductive feature dimension.
            actor_hidden_dims=(128, 128),  # (Backbone) Actor network hidden dimensions.
            value_hidden_dims=(128, 128),  # (Backbone) Value network hidden dimensions.
            layer_norm=True,  # Whether to use layer normalization.
            discount=0.99,  # Discount factor.
            tau=0.005,  # Target network update rate.
            dual_repr_expectile=0.7,  # Dual representation expectile.
            expectile=0.7,  # IQL expectile.
            low_alpha=3.0,  # Low-level AWR temperature.
            high_alpha=3.0,  # High-level AWR temperature.
            subgoal_steps=20,  # Subgoal steps.
            goal_repr_dim=32,  # Subgoal representation dimension.
            dual_repr_dim=256,  # Dual representation latent dimension.
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
            use_bilinear_high_actor=True,  # Whether to use bilinear high-level actor.
            use_backbone=True,  # Whether to use backbone network after transduction.
            value_rep_grad_stop=False,  # Whether to stop gradients from value to goal_repr.
            policy_rep_grad_stop=True,  # Whether to stop gradients from low-level policy to goal_repr (high-level always stops).
        )
    )
    return config