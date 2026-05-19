from typing import Any, Optional, Sequence

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp


def default_init(scale=1.0):
    """Default kernel initializer."""
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={'params': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


class Identity(nn.Module):
    """Identity layer."""

    def __call__(self, x):
        return x


class MLP(nn.Module):
    """Multi-layer perceptron.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        activations: Activation function.
        activate_final: Whether to apply activation to the final layer.
        kernel_init: Kernel initializer.
        layer_norm: Whether to apply layer normalization.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
        return x


class LengthNormalize(nn.Module):
    """Length normalization layer.

    It normalizes the input along the last dimension to have a length of sqrt(dim).
    """

    @nn.compact
    def __call__(self, x):
        return x / (jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-6) * jnp.sqrt(x.shape[-1])


class Param(nn.Module):
    """Scalar parameter module."""

    init_value: float = 0.0

    @nn.compact
    def __call__(self):
        return self.param('value', init_fn=lambda key: jnp.full((), self.init_value))


class LogParam(nn.Module):
    """Scalar parameter module with log scale."""

    init_value: float = 1.0

    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    """Transformed distribution with mode calculation."""

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class RunningMeanStd(flax.struct.PyTreeNode):
    """Running mean and standard deviation.

    Attributes:
        eps: Epsilon value to avoid division by zero.
        mean: Running mean.
        var: Running variance.
        clip_max: Clip value after normalization.
        count: Number of samples.
    """

    eps: Any = 1e-6
    mean: Any = 1.0
    var: Any = 1.0
    clip_max: Any = 10.0
    count: int = 0

    def normalize(self, batch):
        batch = (batch - self.mean) / jnp.sqrt(self.var + self.eps)
        batch = jnp.clip(batch, -self.clip_max, self.clip_max)
        return batch

    def unnormalize(self, batch):
        return batch * jnp.sqrt(self.var + self.eps) + self.mean

    def update(self, batch):
        batch_mean, batch_var = jnp.mean(batch, axis=0), jnp.var(batch, axis=0)
        batch_count = len(batch)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m_2 = m_a + m_b + delta**2 * self.count * batch_count / total_count
        new_var = m_2 / total_count

        return self.replace(mean=new_mean, var=new_var, count=total_count)


class GCActor(nn.Module):
    """Goal-conditioned actor.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        log_std_min: Minimum value of log standard deviation.
        log_std_max: Maximum value of log standard deviation.
        tanh_squash: Whether to squash the action with tanh.
        state_dependent_std: Whether to use state-dependent standard deviation.
        const_std: Whether to use constant standard deviation.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Scaling factor for the standard deviation.
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class GCDiscreteActor(nn.Module):
    """Goal-conditioned actor for discrete actions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        final_fc_init_scale: Initial scale of the final fully-connected layer.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    final_fc_init_scale: float = 1e-2
    gc_encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True)
        self.logit_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))

    def __call__(
        self,
        observations,
        goals=None,
        goal_encoded=False,
        temperature=1.0,
    ):
        """Return the action distribution.

        Args:
            observations: Observations.
            goals: Goals (optional).
            goal_encoded: Whether the goals are already encoded.
            temperature: Inverse scaling factor for the logits (set to 0 to get the argmax).
        """
        if self.gc_encoder is not None:
            inputs = self.gc_encoder(observations, goals, goal_encoded=goal_encoded)
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
            inputs = jnp.concatenate(inputs, axis=-1)
        outputs = self.actor_net(inputs)

        logits = self.logit_net(outputs)

        distribution = distrax.Categorical(logits=logits / jnp.maximum(1e-6, temperature))

        return distribution


class GCValue(nn.Module):
    """Goal-conditioned value/critic function.

    This module can be used for both value V(s, g) and critic Q(s, a, g) functions.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        gc_encoder: Optional GCEncoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    ensemble: bool = True
    gc_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        value_net = mlp_module((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, goals=None, actions=None):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals (optional).
            actions: Actions (optional).
        """
        if self.gc_encoder is not None:
            inputs = [self.gc_encoder(observations, goals)]
        else:
            inputs = [observations]
            if goals is not None:
                inputs.append(goals)
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class GCDiscreteCritic(GCValue):
    """Goal-conditioned critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions)


class GCBilinearValue(nn.Module):
    """Goal-conditioned bilinear value/critic function.

    This module computes the value function as V(s, g) = phi(s)^T psi(g) / sqrt(d) or the critic function as
    Q(s, a, g) = phi(s, a)^T psi(g) / sqrt(d), where phi and psi output d-dimensional vectors.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
        goal_encoder: Optional goal encoder.
        ret_mean: Option to return the mean of phi, psi encodings (useful for using goal representations).
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    ret_mean: bool = False

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
            info: Whether to additionally return the representations phi and psi.
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v)

        if info:
            if self.ensemble and self.ret_mean:
                return v, phi.mean(axis=0), psi.mean(axis=0)
            else:
                return v, phi, psi
        else:
            return v


class GCDiscreteBilinearCritic(GCBilinearValue):
    """Goal-conditioned bilinear critic for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, goals=None, actions=None, info=False):
        actions = jnp.eye(self.action_dim)[actions]
        return super().__call__(observations, goals, actions, info)


class GCMRNValue(nn.Module):
    """Metric residual network (MRN) value function.

    This module computes the value function as the sum of a symmetric Euclidean distance and an asymmetric
    L^infinity-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the MRN value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        sym_s = phi_s[..., : self.latent_dim // 2]
        sym_g = phi_g[..., : self.latent_dim // 2]
        asym_s = phi_s[..., self.latent_dim // 2 :]
        asym_g = phi_g[..., self.latent_dim // 2 :]
        squared_dist = ((sym_s - sym_g) ** 2).sum(axis=-1)
        quasi = jax.nn.relu((asym_s - asym_g).max(axis=-1))
        v = jnp.sqrt(jnp.maximum(squared_dist, 1e-12)) + quasi

        if info:
            return v, phi_s, phi_g
        else:
            return v


class GCIQEValue(nn.Module):
    """Interval quasimetric embedding (IQE) value function.

    This module computes the value function as an IQE-based quasimetric.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        dim_per_component: Dimension of each component in IQE (i.e., number of intervals in each group).
        layer_norm: Whether to apply layer normalization.
        encoder: Optional state/goal encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    dim_per_component: int
    layer_norm: bool = True
    encoder: nn.Module = None

    def setup(self):
        self.phi = MLP((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.alpha = Param()

    def __call__(self, observations, goals, is_phi=False, info=False):
        """Return the IQE value function.

        Args:
            observations: Observations.
            goals: Goals.
            is_phi: Whether the inputs are already encoded by phi.
            info: Whether to additionally return the representations phi_s and phi_g.
        """
        alpha = jax.nn.sigmoid(self.alpha())
        if is_phi:
            phi_s = observations
            phi_g = goals
        else:
            if self.encoder is not None:
                observations = self.encoder(observations)
                goals = self.encoder(goals)
            phi_s = self.phi(observations)
            phi_g = self.phi(goals)

        x = jnp.reshape(phi_s, (*phi_s.shape[:-1], -1, self.dim_per_component))
        y = jnp.reshape(phi_g, (*phi_g.shape[:-1], -1, self.dim_per_component))
        valid = x < y
        xy = jnp.concatenate(jnp.broadcast_arrays(x, y), axis=-1)
        ixy = xy.argsort(axis=-1)
        sxy = jnp.take_along_axis(xy, ixy, axis=-1)
        neg_inc_copies = jnp.take_along_axis(valid, ixy % self.dim_per_component, axis=-1) * jnp.where(
            ixy < self.dim_per_component, -1, 1
        )
        neg_inp_copies = jnp.cumsum(neg_inc_copies, axis=-1)
        neg_f = -1.0 * (neg_inp_copies < 0)
        neg_incf = jnp.concatenate([neg_f[..., :1], neg_f[..., 1:] - neg_f[..., :-1]], axis=-1)
        components = (sxy * neg_incf).sum(axis=-1)
        v = alpha * components.mean(axis=-1) + (1 - alpha) * components.max(axis=-1)

        if info:
            return v, phi_s, phi_g
        else:
            return v


class StateRepresentation(nn.Module):
    """State representation module.
    Attributes:
        hidden_dims: Hidden layer dimensions.
        latent_dim: Latent dimension.
        layer_norm: Whether to apply layer normalization.
        ensemble: Whether to ensemble the value function.
        value_exp: Whether to exponentiate the value. Useful for contrastive learning.
        state_encoder: Optional state encoder.
    """

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None

    def setup(self) -> None:
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)
        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, actions=None, info=False):
        """Return the value/critic function.

        Args:
            observations: Observations.
            goals: Goals.
            actions: Actions (optional).
        """
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)

        return phi


class DiscreteStateActionRepresentation(StateRepresentation):
    """State representation module for discrete actions."""

    action_dim: int = None

    def __call__(self, observations, actions=None, info=False):
        if self.encoder is not None:
            observations = self.encoder(observations)

        if actions is not None:
            actions = jnp.eye(self.action_dim)[actions]

        return super().__call__(observations, actions, info)


class ActorVectorField(nn.Module):
    """Actor vector field network for flow matching.

    Attributes:
        hidden_dims: Hidden layer dimensions.
        action_dim: Action dimension.
        layer_norm: Whether to apply layer normalization.
        encoder: Optional encoder module to encode the inputs.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        """Return the vectors at the given states, actions, and times (optional).

        Args:
            observations: Observations.
            actions: Actions.
            times: Times (optional).
            is_encoded: Whether the observations are already encoded.
        """
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v


class GCHilbertReprValue(nn.Module):
    """Value function parameterized as the Euclidean distance between state & goal representations."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)

    def __call__(self, observations, goals=None):
        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
            if goals is not None and self.goal_encoder is not None:
                goals = self.goal_encoder(goals)
        
        if goals is not None:
            # Value function call.
            state_rep = self.phi(observations)
            goal_rep = self.phi(goals)
            squared_dist = jnp.square(state_rep - goal_rep).sum(axis=-1)
            v = -jnp.sqrt(jnp.maximum(squared_dist, 1e-6))
            return v
        else:
            # Goal encoding; in this case, observations = goals to be encoded.
            phi = self.phi(observations)
            return phi.mean(axis=0) if self.ensemble else phi

class GCBilinearReprValue(nn.Module):
    """Value function parameterized as the inner product between unique state and goal representations."""

    hidden_dims: Sequence[int]
    latent_dim: int
    layer_norm: bool = True
    ensemble: bool = True
    value_exp: bool = False
    state_encoder: nn.Module = None
    goal_encoder: nn.Module = None
    ret_mean: bool = True

    def setup(self):
        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2)

        self.phi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
        self.psi = mlp_module((*self.hidden_dims, self.latent_dim), activate_final=False, layer_norm=self.layer_norm)
    
    def __call__(self, observations, goals=None, actions=None, info: bool = False, goal_encoded: bool = False):
        if goals is None:
            goals = observations
            if self.goal_encoder is not None and not goal_encoded:
                goals = self.goal_encoder(goals)
            psi = self.psi(goals)

            if self.ensemble and self.ret_mean:
                return psi.mean(axis=0)
            return psi

        if self.state_encoder is not None:
            observations = self.state_encoder(observations)
        if self.goal_encoder is not None and not goal_encoded:
            goals = self.goal_encoder(goals)

        if actions is None:
            phi_inputs = observations
        else:
            phi_inputs = jnp.concatenate([observations, actions], axis=-1)

        phi = self.phi(phi_inputs)
        psi = self.psi(goals)

        v = (phi * psi / jnp.sqrt(self.latent_dim)).sum(axis=-1)

        if self.value_exp:
            v = jnp.exp(v) 
        
        if info:
            if self.ensemble and self.ret_mean:
                return v, phi.mean(axis=0), psi.mean(axis=0)
            else:
                return v, phi, psi
        else:
            return v
    

class ProjectedGCBilinearActor(nn.Module):
    latent_dim: int
    action_dim: int
    layer_norm: bool = True
    tr_delta_hidden_dims: Sequence[int] = (128, 128, 128)
    tr_anchor_hidden_dims: Sequence[int] = (128, 128, 128)
    backbone_hidden_dims: Sequence[int] = (128,)
    log_std_min: float = -5
    log_std_max: float = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    anchor_encoder: nn.Module = None
    delta_encoder: nn.Module = None

    def setup(self):
        assert self.tr_delta_hidden_dims[-1] == self.tr_anchor_hidden_dims[-1], \
            "The last hidden dim of delta and anchor modules must be the same."
        assert len(self.tr_delta_hidden_dims) >= 2 and len(self.tr_anchor_hidden_dims) >= 2, \
            "The delta and anchor modules must have at least 2 hidden layers."
        self.bilinear_dim = self.tr_delta_hidden_dims[-1]
        self.fg_output_dim = int(self.bilinear_dim * self.latent_dim)

        self.delta_module = MLP((*self.tr_delta_hidden_dims[:-1], self.fg_output_dim), activate_final=True, layer_norm=self.layer_norm)
        self.anchor_module = MLP((*self.tr_anchor_hidden_dims[:-1], self.fg_output_dim), activate_final=True, layer_norm=self.layer_norm)

        # backbone after transduction
        self.backbone = (
            MLP((*self.backbone_hidden_dims,), activate_final=True, layer_norm=self.layer_norm)
            if len(self.backbone_hidden_dims) > 0 
            else Identity()
        )

        # distribution head (Gaussian)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param("log_stds", nn.initializers.zeros, (self.action_dim,))

    def fg_model(self, delta, anchor):
        # f(delta): (B, T*D) -> (B, T, D)
        f_out = self.delta_module(delta).reshape(delta.shape[0], self.bilinear_dim, self.latent_dim)
        # g(anchor): (B, T*D) -> (B, T, D)
        g_out = self.anchor_module(anchor).reshape(anchor.shape[0], self.bilinear_dim, self.latent_dim)
        # h_t = <f_t, g_t>  -> (B, T)
        h = jnp.sum(f_out * g_out, axis=-1)
        return h

    def __call__(self, delta, anchor, temperature: float = 1.0):
        if self.delta_encoder is not None:
            delta = self.delta_encoder(delta)
        if self.anchor_encoder is not None:
            anchor = self.anchor_encoder(anchor)
        
        squeeze_flag = False
        if delta.ndim == 1:
            delta = delta[None, :]
            squeeze_flag = True
        if anchor.ndim == 1:
            anchor = anchor[None, :]
            squeeze_flag = True

        feature = self.fg_model(delta, anchor)
        feature = self.backbone(feature)
        if squeeze_flag:
            feature = feature.squeeze(0)

        means = self.mean_net(feature)
        if self.state_dependent_std:
            log_stds = self.log_std_net(feature)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        return distribution


class ProjectedGCBilinearValue(nn.Module):
    latent_dim: int
    layer_norm: bool = False
    ensemble: bool = True
    tr_delta_hidden_dims: Sequence[int] = (128, 128, 128)
    tr_anchor_hidden_dims: Sequence[int] = (128, 128, 128)
    backbone_hidden_dims: Sequence[int] = (128,)
    final_fc_init_scale: float = 1e-2
    anchor_encoder: nn.Module = None
    delta_encoder: nn.Module = None

    def setup(self):
        assert self.tr_delta_hidden_dims[-1] == self.tr_anchor_hidden_dims[-1], \
            "The last hidden dim of delta and anchor modules must be the same."
        assert len(self.tr_delta_hidden_dims) >= 2 and len(self.tr_anchor_hidden_dims) >= 2, \
            "The delta and anchor modules must have at least 2 hidden layers."
        self.bilinear_dim = self.tr_delta_hidden_dims[-1]
        self.fg_output_dim = int(self.bilinear_dim * self.latent_dim)

        mlp_module = MLP
        if self.ensemble:
            mlp_module = ensemblize(mlp_module, 2, in_axes=None, out_axes=0)

        self.delta_module = mlp_module((*self.tr_delta_hidden_dims[:-1], self.fg_output_dim), activate_final=True, layer_norm=self.layer_norm)
        self.anchor_module = mlp_module((*self.tr_anchor_hidden_dims[:-1], self.fg_output_dim), activate_final=True, layer_norm=self.layer_norm)

        self.backbone = (
            ensemblize(MLP, 2, in_axes=0, out_axes=0)((*self.backbone_hidden_dims,), activate_final=True, layer_norm=self.layer_norm)
            if (self.ensemble and len(self.backbone_hidden_dims) > 0)
            else MLP((*self.backbone_hidden_dims,), activate_final=True, layer_norm=self.layer_norm)
            if (len(self.backbone_hidden_dims) > 0)
            else Identity()
        )

        self.last = (
            ensemblize(nn.Dense, 2, in_axes=0, out_axes=0)(1, kernel_init=default_init(self.final_fc_init_scale))
            if self.ensemble
            else nn.Dense(1, kernel_init=default_init(self.final_fc_init_scale))
        )

    def fg_model(self, delta, anchor):
        f_out = self.delta_module(delta)
        g_out = self.anchor_module(anchor)

        if self.ensemble:
            f_out = f_out.reshape(f_out.shape[0], f_out.shape[1], self.bilinear_dim, self.latent_dim)
            g_out = g_out.reshape(g_out.shape[0], g_out.shape[1], self.bilinear_dim, self.latent_dim)
        else:
            f_out = f_out.reshape(delta.shape[0], self.bilinear_dim, self.latent_dim)
            g_out = g_out.reshape(anchor.shape[0], self.bilinear_dim, self.latent_dim)

        h = jnp.sum(f_out * g_out, axis=-1)
        return h

    def __call__(self, delta, anchor, actions: Optional[jnp.ndarray] = None):
        if self.delta_encoder is not None:
            delta = self.delta_encoder(delta)
        if self.anchor_encoder is not None:
            anchor = self.anchor_encoder(anchor)

        if actions is not None:
            delta = jnp.concatenate([delta, actions], axis=-1)
            anchor = jnp.concatenate([anchor, actions], axis=-1)

        feature = self.fg_model(delta, anchor)
        feature = self.backbone(feature)
        values = self.last(feature).squeeze(-1)
        return values