import os

import torch as th
import numpy as np

import enum

from . import path
from .utils import EasyDict, log_state, mean_flat
from .integrators import ode, sde, cns_sde


class ModelType(enum.Enum):
    NOISE = enum.auto()
    SCORE = enum.auto()
    VELOCITY = enum.auto()


class PathType(enum.Enum):
    LINEAR = enum.auto()
    GVP = enum.auto()
    VP = enum.auto()


class WeightType(enum.Enum):
    NONE = enum.auto()
    VELOCITY = enum.auto()
    LIKELIHOOD = enum.auto()


class Transport:

    def __init__(self, *, model_type, path_type, loss_type, train_eps, sample_eps):
        path_options = {
            PathType.LINEAR: path.ICPlan,
            PathType.GVP: path.GVPCPlan,
            PathType.VP: path.VPCPlan,
        }
        self.loss_type = loss_type
        self.model_type = model_type
        self.path_sampler = path_options[path_type]()
        self.train_eps = train_eps
        self.sample_eps = sample_eps

    def prior_logp(self, z):
        shape = th.tensor(z.size())
        N = th.prod(shape[1:])
        _fn = lambda x: -N / 2. * np.log(2 * np.pi) - th.sum(x ** 2) / 2.
        return th.vmap(_fn)(z)

    def check_interval(self, train_eps, sample_eps, *, diffusion_form="SBDM",
                       sde=False, reverse=False, eval=False, last_step_size=0.0):
        t0 = 0
        t1 = 1
        eps = train_eps if not eval else sample_eps
        if type(self.path_sampler) in [path.VPCPlan]:
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        elif (type(self.path_sampler) in [path.ICPlan, path.GVPCPlan]
              and (self.model_type != ModelType.VELOCITY or sde)):
            t0 = eps if (diffusion_form == "SBDM" and sde) or self.model_type != ModelType.VELOCITY else 0
            t1 = 1 - eps if (not sde or last_step_size == 0) else 1 - last_step_size
        if reverse:
            t0, t1 = 1 - t0, 1 - t1
        return t0, t1

    def sample(self, x1):
        x0 = th.randn_like(x1)
        t0, t1 = self.check_interval(self.train_eps, self.sample_eps)
        t = th.rand((x1.shape[0],)) * (t1 - t0) + t0
        t = t.to(x1)
        return t, x0, x1

    def training_losses(self, model, x1, model_kwargs=None):
        if model_kwargs is None:
            model_kwargs = {}
        t, x0, x1 = self.sample(x1)
        t, xt, ut = self.path_sampler.plan(t, x0, x1)
        model_output = model(xt, t, **model_kwargs)
        B, *_, C = xt.shape
        assert model_output.size() == (B, *xt.size()[1:-1], C)
        terms = {}
        terms['pred'] = model_output
        if self.model_type == ModelType.VELOCITY:
            terms['loss'] = mean_flat(((model_output - ut) ** 2))
        else:
            _, drift_var = self.path_sampler.compute_drift(xt, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, xt))
            if self.loss_type in [WeightType.VELOCITY]:
                weight = (drift_var / sigma_t) ** 2
            elif self.loss_type in [WeightType.LIKELIHOOD]:
                weight = drift_var / (sigma_t ** 2)
            elif self.loss_type in [WeightType.NONE]:
                weight = 1
            else:
                raise NotImplementedError()
            if self.model_type == ModelType.NOISE:
                terms['loss'] = mean_flat(weight * ((model_output - x0) ** 2))
            else:
                terms['loss'] = mean_flat(weight * ((model_output * sigma_t + x0) ** 2))
        return terms

    def get_drift(self):
        def score_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            model_output = model(x, t, **model_kwargs)
            return -drift_mean + drift_var * model_output

        def noise_ode(x, t, model, **model_kwargs):
            drift_mean, drift_var = self.path_sampler.compute_drift(x, t)
            sigma_t, _ = self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
            model_output = model(x, t, **model_kwargs)
            score = model_output / -sigma_t
            return -drift_mean + drift_var * score

        def velocity_ode(x, t, model, **model_kwargs):
            return model(x, t, **model_kwargs)

        if self.model_type == ModelType.NOISE:
            drift_fn = noise_ode
        elif self.model_type == ModelType.SCORE:
            drift_fn = score_ode
        else:
            drift_fn = velocity_ode

        def body_fn(x, t, model, **model_kwargs):
            model_output = drift_fn(x, t, model, **model_kwargs)
            assert model_output.shape == x.shape, "Output shape from ODE solver must match input shape"
            return model_output

        return body_fn

    def get_score(self):
        if self.model_type == ModelType.NOISE:
            score_fn = lambda x, t, model, **kwargs: (
                model(x, t, **kwargs) / -self.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))[0]
            )
        elif self.model_type == ModelType.SCORE:
            score_fn = lambda x, t, model, **kwargs: model(x, t, **kwargs)
        elif self.model_type == ModelType.VELOCITY:
            score_fn = lambda x, t, model, **kwargs: (
                self.path_sampler.get_score_from_velocity(model(x, t, **kwargs), x, t)
            )
        else:
            raise NotImplementedError()
        return score_fn


class Sampler:
    def __init__(self, transport):
        self.transport = transport
        self.drift = self.transport.get_drift()
        self.score = self.transport.get_score()

    def __get_sde_diffusion_and_drift(self, *, diffusion_form="SBDM", diffusion_norm=1.0):
        def diffusion_fn(x, t):
            return self.transport.path_sampler.compute_diffusion(x, t, form=diffusion_form, norm=diffusion_norm)

        def sde_drift(x, t, model, **kwargs):
            model_output = model(x, t, **kwargs)
            if self.transport.model_type == ModelType.VELOCITY:
                base_drift = model_output
                score = self.transport.path_sampler.get_score_from_velocity(model_output, x, t)
            elif self.transport.model_type == ModelType.NOISE:
                drift_mean, drift_var = self.transport.path_sampler.compute_drift(x, t)
                sigma_t, _ = self.transport.path_sampler.compute_sigma_t(path.expand_t_like_x(t, x))
                score = model_output / -sigma_t
                base_drift = -drift_mean + drift_var * score
            elif self.transport.model_type == ModelType.SCORE:
                drift_mean, drift_var = self.transport.path_sampler.compute_drift(x, t)
                score = model_output
                base_drift = -drift_mean + drift_var * score
            else:
                raise NotImplementedError()
            return base_drift + diffusion_fn(x, t) * score

        return sde_drift, diffusion_fn

    def __get_last_step(self, sde_drift, *, last_step, last_step_size):
        if last_step is None:
            return lambda x, t, model, **kw: x
        elif last_step == "Mean":
            return lambda x, t, model, **kw: x + sde_drift(x, t, model, **kw) * last_step_size
        elif last_step == "Tweedie":
            alpha = self.transport.path_sampler.compute_alpha_t
            sigma = self.transport.path_sampler.compute_sigma_t
            return lambda x, t, model, **kw: (
                x / alpha(t)[0][0] + (sigma(t)[0][0] ** 2) / alpha(t)[0][0] * self.score(x, t, model, **kw)
            )
        elif last_step == "Euler":
            return lambda x, t, model, **kw: x + self.drift(x, t, model, **kw) * last_step_size
        else:
            raise NotImplementedError()

    def sample_sde(
        self,
        *,
        sampling_method="Euler",
        diffusion_form="SBDM",
        diffusion_norm=1.0,
        last_step="Mean",
        last_step_size=0.04,
        num_steps=250,
        cns=False,
        gamma_matrix_path="gamma_matrix/gamma_matrix_scaled.pt",
        gamma_matrix_divider=1.0,
        sqrt_gamma=False,
        power_gamma=1.0,
        alpha_tilting=0.0,
        alpha_tilting_inside_exp=False,
        alpha_tilting_use_fnorm=False,
        alpha_exponential_interpolation=False,
        alpha_exponential_interpolation_sharpness=4.0,
        energy_scale=1.0,
        return_velocity=False,
    ):
        if last_step is None:
            last_step_size = 0.0

        sde_drift, sde_diffusion = self.__get_sde_diffusion_and_drift(
            diffusion_form=diffusion_form, diffusion_norm=diffusion_norm,
        )
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            diffusion_form=diffusion_form, sde=True, eval=True,
            reverse=False, last_step_size=last_step_size,
        )

        if not cns:
            _sde = sde(
                sde_drift, sde_diffusion,
                t0=t0, t1=t1, num_steps=num_steps, sampler_type=sampling_method,
            )
        else:
            _sde = cns_sde(
                sde_drift, sde_diffusion,
                t0=t0, t1=t1, num_steps=num_steps, sampler_type=sampling_method,
                gamma_matrix=th.load(gamma_matrix_path, map_location="cpu") if os.path.exists(gamma_matrix_path) else None,
                gamma_matrix_divider=gamma_matrix_divider,
                sqrt_gamma=sqrt_gamma,
                power_gamma=power_gamma,
                alpha_tilting=alpha_tilting,
                alpha_tilting_inside_exp=alpha_tilting_inside_exp,
                alpha_tilting_use_fnorm=alpha_tilting_use_fnorm,
                alpha_exponential_interpolation=alpha_exponential_interpolation,
                alpha_exponential_interpolation_sharpness=alpha_exponential_interpolation_sharpness,
                energy_scale=energy_scale,
                return_velocity=return_velocity,
            )

        last_step_fn = self.__get_last_step(sde_drift, last_step=last_step, last_step_size=last_step_size)

        def _sample(init, model, **model_kwargs):
            if not return_velocity:
                xs = _sde.sample(init, model, **model_kwargs)
            else:
                xs, velocities = _sde.sample(init, model, **model_kwargs)
            ts = th.ones(init.size(0), device=init.device) * t1
            x = last_step_fn(xs[-1], ts, model, **model_kwargs)
            xs.append(x)
            assert len(xs) == num_steps, f"Sample count mismatch: {len(xs)} vs {num_steps}"
            if not return_velocity:
                return xs
            return xs, velocities, init

        return _sample, _sde

    def sample_ode(self, *, sampling_method="dopri5", num_steps=50,
                   atol=1e-6, rtol=1e-3, reverse=False):
        drift = self.drift
        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=reverse, last_step_size=0.0,
        )
        _ode = ode(
            drift=drift, t0=t0, t1=t1,
            sampler_type=sampling_method, num_steps=num_steps, atol=atol, rtol=rtol,
        )
        return _ode.sample, _ode

    def sample_ode_likelihood(self, *, sampling_method="dopri5", num_steps=50,
                              atol=1e-6, rtol=1e-3):
        def _likelihood_drift(x, t, model, **model_kwargs):
            x, _ = x
            eps = th.randint(2, x.size(), dtype=th.float, device=x.device) * 2 - 1
            t = th.ones_like(t) * (1 - t)
            with th.enable_grad():
                x.requires_grad = True
                grad = th.autograd.grad(th.sum(self.drift(x, t, model, **model_kwargs) * eps), x)[0]
                logp_grad = th.sum(grad * eps, dim=tuple(range(1, len(x.size()))))
                drift = self.drift(x, t, model, **model_kwargs)
            return -drift, logp_grad

        t0, t1 = self.transport.check_interval(
            self.transport.train_eps, self.transport.sample_eps,
            sde=False, eval=True, reverse=False, last_step_size=0.0,
        )
        _ode = ode(
            drift=_likelihood_drift, t0=t0, t1=t1,
            sampler_type=sampling_method, num_steps=num_steps, atol=atol, rtol=rtol,
        )

        def _sample_fn(x, model, **model_kwargs):
            init_logp = th.zeros(x.size(0)).to(x)
            drift, delta_logp = _ode.sample((x, init_logp), model, **model_kwargs)
            drift, delta_logp = drift[-1], delta_logp[-1]
            prior_logp = self.transport.prior_logp(drift)
            return prior_logp - delta_logp, drift

        return _sample_fn
