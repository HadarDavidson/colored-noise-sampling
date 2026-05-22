# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Samples a large number of images from a pre-trained SiT model using DDP.
Subsequently saves a .npz file that can be used to compute FID and other
evaluation metrics via the ADM repo:
  https://github.com/openai/guided-diffusion/tree/main/evaluations

For a simple single-GPU/CPU sampling script, see sample.py.
"""
import torch
import torch.distributed as dist
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_ode_args, parse_sde_args, parse_transport_args
from tqdm import tqdm
from PIL import Image
import numpy as np
import math
import argparse
import sys
import datetime
import os


class FrequencyAnalyzer:
    """DDP-aware spectral progress analyzer for ODE trajectories.

    Accumulates the per-frequency-bin image-completion progress matrix online
    (Welford-style) and signals convergence when the 99.9 % confidence interval
    stops improving.  The saved mean matrix is the gamma matrix used by CNS.
    """

    def __init__(self, num_timesteps, num_bins=32, device="cuda"):
        self.num_bins = num_bins
        self.device = device
        self.num_timesteps = num_timesteps
        self.count = torch.tensor(0.0, device=device)
        self.sum_gamma = torch.zeros(num_timesteps, num_bins, device=device)
        self.sum_sq_gamma = torch.zeros(num_timesteps, num_bins, device=device)
        self._freq_indices = None

    def _init_grid(self, H, W):
        freq_y = torch.fft.fftfreq(H, device=self.device).view(1, 1, H, 1)
        freq_x = torch.fft.fftfreq(W, device=self.device).view(1, 1, 1, W)
        r = torch.sqrt(freq_x ** 2 + freq_y ** 2)
        r_norm = r / r.max()
        self._freq_indices = (r_norm * (self.num_bins - 1)).long().clamp(0, self.num_bins - 1)

    def update(self, trajectory):
        """Process one ODE trajectory batch of shape (T, B, C, H, W)."""
        if isinstance(trajectory, list):
            trajectory = torch.stack(trajectory, dim=0)
        T, B, C, H, W = trajectory.shape
        if self._freq_indices is None:
            self._init_grid(H, W)

        # Reconstruct clean prediction at every step via the velocity estimate.
        # SiT: t=0 is noise, t=1 is data.  x_pred = x_t + (1-t) * v_t
        t_seq = torch.linspace(0.0, 1.0, T, device=self.device).view(-1, 1, 1, 1, 1)
        dt = 1.0 / (T - 1)
        v_t = (trajectory[1:] - trajectory[:-1]) / dt
        x_pred = trajectory[:-1] + (1.0 - t_seq[:-1]) * v_t
        # Anchor: final generated image
        x_pred = torch.cat([x_pred, trajectory[-1:]], dim=0)  # (T, B, C, H, W)

        # Per-frequency image-completion progress
        traj_fft = torch.fft.fft2(x_pred.to(torch.float32))
        X_final = traj_fft[-1].unsqueeze(0)
        error_energy = (traj_fft - X_final).abs() ** 2
        signal_energy = X_final.abs() ** 2 + 1e-8
        progress = (1.0 - error_energy / signal_energy).clamp(0, 1)
        progress = progress.mean(dim=2)  # (T, B, H, W) — average over channels

        # Radial binning
        binned = torch.zeros(T, B, self.num_bins, device=self.device)
        binned.scatter_add_(
            2,
            self._freq_indices.view(-1).unsqueeze(0).unsqueeze(0).expand(T, B, -1),
            progress.view(T, B, -1),
        )
        bin_counts = torch.bincount(self._freq_indices.view(-1), minlength=self.num_bins).float()
        binned = binned / bin_counts.unsqueeze(0).unsqueeze(0).clamp(min=1)

        # DDP reduce
        batch_sum = binned.sum(dim=1)
        batch_sq_sum = (binned ** 2).sum(dim=1)
        batch_count = torch.tensor(B, dtype=torch.float32, device=self.device)
        dist.all_reduce(batch_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_sq_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(batch_count, op=dist.ReduceOp.SUM)

        self.sum_gamma += batch_sum
        self.sum_sq_gamma += batch_sq_sum
        self.count += batch_count

    def get_convergence_stats(self, lookahead_samples=1000, target_improvement=1e-4):
        """Return (mean_matrix, max_CI_error, max_improvement, est_remaining_samples)."""
        mean = self.sum_gamma / self.count
        raw_var = (self.sum_sq_gamma - (self.sum_gamma ** 2) / self.count) / (self.count - 1 + 1e-8)
        var = torch.clamp(raw_var, min=0.0)
        z = 3.291  # 99.9 % CI
        cur_err = z * torch.sqrt(var / self.count)
        proj_err = z * torch.sqrt(var / (self.count + lookahead_samples))
        improvement = cur_err - proj_err
        max_err = cur_err.max().item()
        max_imp = improvement.max().item()
        est_remaining = 0
        if max_imp > 0:
            n_cur = self.count.item()
            est_remaining = max(0, int(n_cur * ((max_imp / target_improvement) ** (2 / 3)) - n_cur))
        return mean, max_err, max_imp, est_remaining


def create_npz_from_sample_folder(sample_dir, num=50_000):
    samples = []
    for i in tqdm(range(num), desc="Building .npz file from samples"):
        sample_pil = Image.open(f"{sample_dir}/{i:06d}.png")
        sample_np = np.asarray(sample_pil).astype(np.uint8)
        samples.append(sample_np)
    samples = np.stack(samples)
    assert samples.shape == (num, samples.shape[1], samples.shape[2], 3)
    timestamp = datetime.datetime.now().strftime("%H:%M:%S_%d-%m-%Y")
    npz_path = f"{sample_dir}_{timestamp}.npz"
    np.savez(npz_path, arr_0=samples)
    print(f"Saved .npz file to {npz_path} [shape={samples.shape}].")
    return npz_path


def main(mode, args):
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    assert torch.cuda.is_available(), "Sampling with DDP requires at least one GPU."
    torch.set_grad_enabled(False)

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    if args.ckpt is None:
        assert args.model == "SiT-XL/2", "Only SiT-XL/2 models are available for auto-download."
        assert args.image_size in [256, 512]
        assert args.num_classes == 1000
        assert args.image_size == 256, "512x512 models are not yet available for auto-download."
        learn_sigma = args.image_size == 256
    else:
        learn_sigma = False

    latent_size = args.image_size // 8
    dtype_model = torch.float32
    if args.dtype == "float16":
        dtype_model = torch.float16
        torch.backends.cuda.matmul.allow_tf32 = False
    elif args.dtype == "bfloat16":
        dtype_model = torch.bfloat16

    model = SiT_models[args.model](
        input_size=latent_size,
        num_classes=args.num_classes,
        learn_sigma=learn_sigma,
    ).to(device, dtype=dtype_model)
    ckpt_path = args.ckpt or f"SiT-XL-2-{args.image_size}x{args.image_size}.pt"
    state_dict = find_model(ckpt_path)
    model.load_state_dict(state_dict)
    model.eval()
    model = torch.compile(model)

    transport = create_transport(args.path_type, args.prediction, args.loss_weight,
                                 args.train_eps, args.sample_eps)
    sampler = Sampler(transport)

    if mode == "ODE":
        if args.likelihood:
            assert args.cfg_scale == 1, "Likelihood is incompatible with guidance"
            sample_fn = sampler.sample_ode_likelihood(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol, rtol=args.rtol,
            )
        else:
            sample_fn, _ = sampler.sample_ode(
                sampling_method=args.sampling_method,
                num_steps=args.num_sampling_steps,
                atol=args.atol, rtol=args.rtol,
                reverse=args.reverse,
            )
    elif mode == "SDE":
        if len(args.alpha_tilting) == 1:
            alpha_tilting_param = args.alpha_tilting[0]
        elif len(args.alpha_tilting) == 2:
            alpha_tilting_param = args.alpha_tilting
        else:
            raise ValueError(f"--alpha-tilting expects 1 or 2 arguments, got {len(args.alpha_tilting)}")

        sample_fn, _ = sampler.sample_sde(
            sampling_method=args.sampling_method,
            diffusion_form=args.diffusion_form,
            diffusion_norm=args.diffusion_norm,
            last_step=args.last_step,
            last_step_size=args.last_step_size,
            num_steps=args.num_sampling_steps,
            cns=args.cns,
            gamma_matrix_path=args.gamma_matrix_path,
            gamma_matrix_divider=args.gamma_matrix_divider,
            sqrt_gamma=args.sqrt_gamma,
            power_gamma=args.power_gamma,
            alpha_tilting=alpha_tilting_param,
            alpha_tilting_inside_exp=args.alpha_tilting_inside_exp,
            alpha_tilting_use_fnorm=args.alpha_tilting_use_fnorm,
            alpha_exponential_interpolation=args.alpha_exponential_interpolation,
            alpha_exponential_interpolation_sharpness=args.alpha_exponential_interpolation_sharpness,
            energy_scale=args.energy_scale,
        )

    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    vae = torch.compile(vae)
    assert args.cfg_scale >= 1.0, "cfg_scale must be >= 1.0"
    using_cfg = args.cfg_scale > 1.0

    # Build output folder name
    model_string_name = args.model.replace("/", "-")
    ckpt_string_name = os.path.basename(args.ckpt).replace(".pt", "") if args.ckpt else "pretrained"
    if mode == "ODE":
        folder_name = (f"{model_string_name}-{ckpt_string_name}-"
                       f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"
                       f"{mode}-{args.num_sampling_steps}-{args.sampling_method}"
                       f"seed-{args.global_seed}")
    elif mode == "SDE":
        folder_name = (f"{model_string_name}-{ckpt_string_name}-"
                       f"cfg-{args.cfg_scale}-{args.per_proc_batch_size}-"
                       f"{mode}-{args.num_sampling_steps}-{args.sampling_method}-"
                       f"{args.diffusion_form}-{args.last_step}-{args.last_step_size}"
                       f"seed-{args.global_seed}")
        if args.cns:
            folder_name += "-cns"
            if args.sqrt_gamma:
                folder_name += "-sqrt-gamma"
            elif args.power_gamma != 1.0:
                folder_name += f"-power-gamma-{args.power_gamma}"
            if args.gamma_matrix_divider != 1.0:
                folder_name += f"-gamma-divider-{args.gamma_matrix_divider}"
            if args.gamma_matrix_path != "gamma_matrix/gamma_matrix_smoothed_scaled.pt":
                if "smoothed" in args.gamma_matrix_path:
                    gamma_matrix_number = (args.gamma_matrix_path.split("_")[-1].split(".")[0]
                                           + "." + args.gamma_matrix_path.split("_")[-1].split(".")[1])
                    folder_name += f"-sigma-smooth-{gamma_matrix_number}"
                else:
                    folder_name += "-non-smoothed"
                    if "250steps" in args.gamma_matrix_path:
                        folder_name += "-non-scaled"
                    if "sde" in args.gamma_matrix_path or "SDE" in args.gamma_matrix_path:
                        folder_name += "-sde-matrix"
                    if "global" in args.gamma_matrix_path:
                        folder_name += "-global-scaled"
                    if "v2" in args.gamma_matrix_path:
                        folder_name += "-v2"
                    if "v3" in args.gamma_matrix_path:
                        folder_name += "-v3"

        if args.alpha_tilting != 0.0 and args.alpha_tilting != [0.0]:
            folder_name += f"-alpha-{args.alpha_tilting}"
            if not args.alpha_tilting_inside_exp and args.cns:
                folder_name += "-outside-exp"
            if args.alpha_tilting_use_fnorm and args.cns:
                folder_name += "-fnorm-tilting"
            if args.alpha_exponential_interpolation and args.cns:
                folder_name += f"-exp-interp{args.alpha_exponential_interpolation_sharpness}"

    if args.path_type != "Linear":
        folder_name += f"-path-{args.path_type}"
    if args.energy_scale != 1.0:
        folder_name += f"-energy-scale{args.energy_scale}"
    if args.per_iter_seed:
        folder_name += "-per-iter-seed"

    sample_folder_dir = f"{args.sample_dir}/{folder_name}"
    if rank == 0:
        os.makedirs(sample_folder_dir, exist_ok=True)
        print(f"Saving .png samples at {sample_folder_dir}")
    dist.barrier()

    n = args.per_proc_batch_size
    global_batch_size = n * dist.get_world_size()

    if args.analyze_spectrum:
        assert mode == "ODE", "--analyze-spectrum requires ODE mode to compute clean gamma matrices."
        if rank == 0:
            print(f"=== Starting Spectral Analysis ===")
            print(f"Minimum samples before convergence check: {args.min_spectrum_samples}")
        analyzer = FrequencyAnalyzer(args.num_sampling_steps, num_bins=32, device=device)
        iterations = 999_999  # broken by convergence check below
    else:
        total_samples = int(math.ceil(args.num_fid_samples / global_batch_size) * global_batch_size)
        iterations = int(total_samples // global_batch_size)
        if rank == 0:
            print(f"Total number of images that will be sampled: {total_samples}")
            assert total_samples % dist.get_world_size() == 0
            samples_needed_this_gpu = int(total_samples // dist.get_world_size())
            assert samples_needed_this_gpu % n == 0
            iterations = int(samples_needed_this_gpu // n)

    pbar = range(iterations)
    total = 0
    pbar = tqdm(pbar) if rank == 0 else pbar

    for i in pbar:
        if args.per_iter_seed:
            iter_seed = (args.global_seed * 1_000_000 + i) * dist.get_world_size() + rank
            _gen = torch.Generator(device=device)
            _gen.manual_seed(iter_seed)
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device, generator=_gen)
            y = torch.randint(0, args.num_classes, (n,), device=device, generator=_gen)
        else:
            z = torch.randn(n, model.in_channels, latent_size, latent_size, device=device)
            y = torch.randint(0, args.num_classes, (n,), device=device)

        if dtype_model in [torch.float16, torch.bfloat16]:
            z = z.to(dtype_model)

        if using_cfg:
            z = torch.cat([z, z], 0)
            y_null = torch.tensor([1000] * n, device=device)
            y = torch.cat([y, y_null], 0)
            model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
            model_fn = model.forward_with_cfg
        else:
            model_kwargs = dict(y=y)
            model_fn = model.forward

        with torch.autocast(device_type="cuda", dtype=dtype_model):
            traj = sample_fn(z, model_fn, **model_kwargs)

        # ---------------------------------------------------------------
        # SPECTRAL ANALYSIS PATH  (ODE gamma-matrix computation)
        # ---------------------------------------------------------------
        if args.analyze_spectrum:
            traj_tensor = traj if not isinstance(traj, list) else torch.stack(traj, dim=0)
            if using_cfg:
                traj_tensor, _ = traj_tensor.chunk(2, dim=1)

            analyzer.update(traj_tensor)

            current_count = int(analyzer.count.item())
            lookahead = global_batch_size * 5
            target_imp = 1e-4
            should_stop = torch.tensor(0, device=device)

            if current_count >= args.min_spectrum_samples:
                mean_matrix, max_error, max_improvement, est_remaining = analyzer.get_convergence_stats(
                    lookahead_samples=lookahead, target_improvement=target_imp
                )
                if max_improvement < target_imp:
                    should_stop.fill_(1)

            dist.all_reduce(should_stop, op=dist.ReduceOp.MAX)

            if rank == 0:
                if current_count >= args.min_spectrum_samples:
                    pbar.set_description(
                        f"Samples: {current_count} | CI: ±{max_error:.4f} | "
                        f"Gain: {max_improvement:.6f} | Est. remaining: {est_remaining}"
                    )
                else:
                    pbar.set_description(
                        f"Samples: {current_count} / {args.min_spectrum_samples} (warming up)"
                    )

            if should_stop.item() > 0:
                if rank == 0:
                    print(f"\n=== Convergence reached! Final CI: ±{max_error:.4f} ===")
                    mean_matrix, _, _, _ = analyzer.get_convergence_stats()
                    cfg_suffix = f"_cfg_{args.cfg_scale}" if using_cfg else ""
                    save_path = (
                        f"{sample_folder_dir}/gamma_matrix_"
                        f"{args.num_sampling_steps}steps_ODE{cfg_suffix}.pt"
                    )
                    torch.save(mean_matrix.cpu(), save_path)
                    print(f"Gamma matrix saved to: {save_path}")
                break

            continue
        # ---------------------------------------------------------------

        samples = traj[-1]
        if using_cfg:
            samples, _ = samples.chunk(2, dim=0)

        samples = vae.decode(samples.to(vae.dtype) / 0.18215).sample
        samples = torch.clamp(127.5 * samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()

        for j, sample in enumerate(samples):
            index = j * dist.get_world_size() + rank + total
            Image.fromarray(sample).save(f"{sample_folder_dir}/{index:06d}.png")

        total += global_batch_size
        dist.barrier()

    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and not args.analyze_spectrum:
        create_npz_from_sample_folder(sample_folder_dir, args.num_fid_samples)
        print("Done.")


if __name__ == "__main__":
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(False)
    parser = argparse.ArgumentParser()

    if len(sys.argv) < 2:
        print("Usage: program.py <mode> [options]")
        sys.exit(1)

    mode = sys.argv[1]
    assert mode[:2] != "--", "Usage: program.py <mode> [options]"
    assert mode in ["ODE", "SDE"], "Invalid mode. Choose 'ODE' or 'SDE'"

    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")
    parser.add_argument("--sample-dir", type=str, default="samples")
    parser.add_argument("--per-proc-batch-size", type=int, default=4)
    parser.add_argument("--num-fid-samples", type=int, default=50_000)
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--per-iter-seed", action=argparse.BooleanOptionalAction, default=False,
                        help="Seed each iteration independently for reproducible ODE/SDE/CNS comparison.")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to SiT checkpoint (default: auto-download SiT-XL/2).")
    parser.add_argument("--dtype", type=str, default="float32",
                        choices=["float32", "float16", "bfloat16"])

    # CNS (Colored Noise Sampling) arguments
    parser.add_argument("--cns", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable CNS (Colored Noise Sampling) (SDE mode only).")
    parser.add_argument("--gamma-matrix-path", type=str,
                        default="gamma_matrix/gamma_matrix_scaled.pt",
                        help="Path to the pre-computed DyPE gamma matrix.")
    parser.add_argument("--gamma-matrix-divider", type=float, default=1.0,
                        help="Divider applied to gamma matrix values (controls noise residual scale).")
    parser.add_argument("--sqrt-gamma", action=argparse.BooleanOptionalAction, default=False,
                        help="Apply sqrt to residual energy (amplitude vs. energy scaling).")
    parser.add_argument("--power-gamma", type=float, default=1.0,
                        help="Power applied to residual energy.")
    parser.add_argument("--alpha-tilting", type=float, nargs='+', default=[0.0],
                        help="Frequency tilt: one float (constant) or two floats (start end) for time-varying.")
    parser.add_argument("--alpha-tilting-inside-exp", action=argparse.BooleanOptionalAction, default=False,
                        help="Place colored noise residual inside exponent: exp(alpha * f * residual).")
    parser.add_argument("--alpha-tilting-use-fnorm", action=argparse.BooleanOptionalAction, default=False,
                        help="Guide tilt by normalized frequency position.")
    parser.add_argument("--alpha-exponential-interpolation", action=argparse.BooleanOptionalAction, default=False,
                        help="Use exponential (vs. linear) interpolation for time-varying alpha.")
    parser.add_argument("--alpha-exponential-interpolation-sharpness", type=float, default=4.0,
                        help="Sharpness of exponential alpha interpolation.")
    parser.add_argument("--energy-scale", type=float, default=1.0,
                        help="Scale factor on CNS noise std after unit-std normalization.")

    # Gamma-matrix computation via ODE spectral analysis
    parser.add_argument("--analyze-spectrum", action="store_true",
                        help="Run ODE frequency analysis to build the gamma matrix. "
                             "Iterates until the 99.9%% CI converges, then saves the matrix "
                             "inside the sample folder and exits. Requires ODE mode.")
    parser.add_argument("--min-spectrum-samples", type=int, default=4096,
                        help="Minimum number of samples to accumulate before the convergence "
                             "check starts. Default 4096.")

    parse_transport_args(parser)
    if mode == "ODE":
        parse_ode_args(parser)
    elif mode == "SDE":
        parse_sde_args(parser)

    args = parser.parse_known_args()[0]
    main(mode, args)
