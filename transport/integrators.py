import math
import numpy as np
import torch as th
from torchdiffeq import odeint


class sde:
    """Standard Euler-Maruyama / Heun SDE solver."""
    def __init__(self, drift, diffusion, *, t0, t1, num_steps, sampler_type):
        assert t0 < t1
        self.t0 = t0
        self.t1 = t1
        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type

    def __Euler_Maruyama_step(self, x, mean_x, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        t = th.ones(x.size(0)).to(x) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t, model, **model_kwargs)
        diffusion = self.diffusion(x, t)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x

    def __Heun_step(self, x, _, t, model, **model_kwargs):
        w_cur = th.randn(x.size()).to(x)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0)).to(x) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat

    def __forward_fn(self):
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }
        try:
            return sampler_dict[self.sampler_type]
        except KeyError:
            raise NotImplementedError(f"Sampler type '{self.sampler_type}' not implemented.")

    def sample(self, init, model, **model_kwargs):
        x = init
        mean_x = init
        samples = []
        sampler = self.__forward_fn()
        for ti in self.t[:-1]:
            with th.no_grad():
                x, mean_x = sampler(x, mean_x, ti, model, **model_kwargs)
                samples.append(x)
        return samples


class ode:
    """ODE solver using torchdiffeq."""
    def __init__(self, drift, *, t0, t1, sampler_type, num_steps, atol, rtol):
        self.drift = drift
        self.t = th.linspace(t0, t1, num_steps)
        self.atol = atol
        self.rtol = rtol
        self.sampler_type = sampler_type

    def sample(self, x, model, **model_kwargs):
        device = x[0].device if isinstance(x, tuple) else x.device

        def _fn(t, x):
            t = (th.ones(x[0].size(0)).to(device) * t
                 if isinstance(x, tuple)
                 else th.ones(x.size(0)).to(device) * t)
            return self.drift(x, t, model, **model_kwargs)

        t = self.t.to(device)
        atol = [self.atol] * len(x) if isinstance(x, tuple) else [self.atol]
        rtol = [self.rtol] * len(x) if isinstance(x, tuple) else [self.rtol]
        return odeint(_fn, x, t, method=self.sampler_type, atol=atol, rtol=rtol)


class cns_sde:
    """CNS (Colored Noise Sampling) SDE with empirical DyPE gamma-matrix noise shaping.

    At each step the isotropic white noise of a standard SDE is replaced by
    spectrally shaped noise whose per-frequency amplitude follows the residual
    energy profile derived from the empirical DyPE gamma matrix.
    """

    def __init__(
        self,
        drift,
        diffusion,
        *,
        t0,
        t1,
        num_steps,
        sampler_type,
        gamma_matrix,
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
        assert t0 < t1
        self.t0 = t0
        self.t1 = t1
        self.num_timesteps = num_steps
        self.t = th.linspace(t0, t1, num_steps)
        self.dt = self.t[1] - self.t[0]
        self.drift = drift
        self.diffusion = diffusion
        self.sampler_type = sampler_type
        self.gamma_matrix = gamma_matrix
        self.num_freq_bins = gamma_matrix.size(1)
        self.gamma_matrix_divider = gamma_matrix_divider
        self.sqrt_gamma = sqrt_gamma
        self.power_gamma = power_gamma
        self.alpha_tilting = alpha_tilting
        self.alpha_tilting_inside_exp = alpha_tilting_inside_exp
        self.alpha_tilting_use_fnorm = alpha_tilting_use_fnorm
        self.alpha_exponential_interpolation = alpha_exponential_interpolation
        self.alpha_exponential_interpolation_sharpness = alpha_exponential_interpolation_sharpness
        self.energy_scale = energy_scale
        self.return_velocity = return_velocity

    def __generate_matrix_scaled_noise(self, x, step_idx):
        """Returns unit-std spectrally-shaped noise for this sampling step."""
        w = th.randn(x.size(), device=x.device, dtype=x.dtype)
        if w.dim() != 4:
            return w
        B, C, H, W = w.size()

        gamma_row = self.gamma_matrix[step_idx].to(x.device)

        # Time-varying alpha: linear or exponential interpolation between [alpha_start, alpha_end]
        if isinstance(self.alpha_tilting, (list, tuple)) and len(self.alpha_tilting) == 2:
            progress = step_idx / max(1, self.num_timesteps - 1)
            alpha_start, alpha_end = self.alpha_tilting
            if not self.alpha_exponential_interpolation:
                current_alpha = alpha_start + progress * (alpha_end - alpha_start)
            else:
                sharpness = self.alpha_exponential_interpolation_sharpness
                exp_progress = (math.exp(sharpness * progress) - 1.0) / (math.exp(sharpness) - 1.0)
                current_alpha = alpha_start + exp_progress * (alpha_end - alpha_start)
        else:
            current_alpha = float(self.alpha_tilting)

        f_norm = th.linspace(0.0, 1.0, steps=self.num_freq_bins, device=x.device)
        base_residual = 1.0 - gamma_row / self.gamma_matrix_divider

        if current_alpha != 0.0:
            if self.alpha_tilting_use_fnorm:
                tilt_base = f_norm
            elif self.alpha_tilting_inside_exp:
                tilt_base = 1.0
            else:
                raise ValueError(
                    "alpha_tilting_use_fnorm=False requires alpha_tilting_inside_exp=True; "
                    "otherwise no frequency-dependent tilt is applied."
                )
            if self.alpha_tilting_inside_exp:
                residual_energy = th.exp(current_alpha * tilt_base * base_residual)
            else:
                residual_energy = th.exp(current_alpha * tilt_base) * base_residual
        else:
            residual_energy = base_residual.clamp(min=0.0)

        if self.sqrt_gamma:
            noise_scaling = th.sqrt(residual_energy.clamp(min=0.0))
        elif self.power_gamma != 1.0:
            noise_scaling = residual_energy.clamp(min=0.0) ** self.power_gamma
        else:
            noise_scaling = residual_energy.clamp(min=0.0)

        # Cache the 2D radial frequency-bin index map (recomputed only on spatial-dim change)
        if not hasattr(self, '_freq_indices') or self._freq_indices.shape[-2:] != (H, W):
            freq_y = th.fft.fftfreq(H, device=x.device).view(1, 1, H, 1)
            freq_x_v = th.fft.fftfreq(W, device=x.device).view(1, 1, 1, W)
            r = th.sqrt(freq_x_v ** 2 + freq_y ** 2)
            r_norm = r / r.max()
            self._freq_indices = (r_norm * (self.num_freq_bins - 1)).long().clamp(0, self.num_freq_bins - 1)

        scaling_grid = noise_scaling[self._freq_indices]
        w_filtered = th.fft.ifft2(th.fft.fft2(w.to(th.float32)) * scaling_grid).real

        # Renormalize to unit std to preserve total energy
        w_std = w_filtered.std()
        if w_std > 1e-9:
            w_filtered = w_filtered / w_std

        if self.energy_scale != 1.0:
            w_filtered = w_filtered * self.energy_scale

        # Diagnostic: print noise energy and per-band scaling at key steps (rank-0 only)
        is_rank0 = not th.distributed.is_initialized() or th.distributed.get_rank() == 0
        if is_rank0 and (step_idx < 5 or step_idx % 50 == 0 or step_idx == self.num_timesteps - 2):
            with th.no_grad():
                energy_ratio = w_filtered.var().item() / max(w.var().item(), 1e-9)
                s_low  = noise_scaling[0].item()
                s_mid  = noise_scaling[self.num_freq_bins // 2].item()
                s_high = noise_scaling[-1].item()
                print(f"[CNS step {step_idx:03d}] energy={energy_ratio:.4f} | "
                      f"scale L/M/H: {s_low:.3f}/{s_mid:.3f}/{s_high:.3f}")

        return w_filtered.to(x.dtype)

    def __Euler_Maruyama_step(self, x, mean_x, t, step_idx, model, **model_kwargs):
        w_cur = self.__generate_matrix_scaled_noise(x, step_idx)
        t_tensor = th.ones(x.size(0), device=x.device, dtype=x.dtype) * t
        dw = w_cur * th.sqrt(self.dt)
        drift = self.drift(x, t_tensor, model, **model_kwargs)
        diffusion = self.diffusion(x, t_tensor)
        mean_x = x + drift * self.dt
        x = mean_x + th.sqrt(2 * diffusion) * dw
        return x, mean_x, drift

    def __Heun_step(self, x, _, t, step_idx, model, **model_kwargs):
        w_cur = self.__generate_matrix_scaled_noise(x, step_idx)
        dw = w_cur * th.sqrt(self.dt)
        t_cur = th.ones(x.size(0), device=x.device, dtype=x.dtype) * t
        diffusion = self.diffusion(x, t_cur)
        xhat = x + th.sqrt(2 * diffusion) * dw
        K1 = self.drift(xhat, t_cur, model, **model_kwargs)
        xp = xhat + self.dt * K1
        K2 = self.drift(xp, t_cur + self.dt, model, **model_kwargs)
        return xhat + 0.5 * self.dt * (K1 + K2), xhat, K1

    def __forward_fn(self):
        sampler_dict = {
            "Euler": self.__Euler_Maruyama_step,
            "Heun": self.__Heun_step,
        }
        try:
            return sampler_dict[self.sampler_type]
        except KeyError:
            raise NotImplementedError(f"Sampler type '{self.sampler_type}' not implemented for CNS.")

    def sample(self, init, model, **model_kwargs):
        x = init
        mean_x = init
        samples = []
        velocities = [] if self.return_velocity else None
        sampler = self.__forward_fn()

        for step_idx, ti in enumerate(self.t[:-1]):
            with th.inference_mode():
                x, mean_x, drift = sampler(x, mean_x, ti, step_idx, model, **model_kwargs)
                samples.append(x)
                if self.return_velocity:
                    velocities.append(drift)

        if not self.return_velocity:
            return samples
        return samples, velocities
