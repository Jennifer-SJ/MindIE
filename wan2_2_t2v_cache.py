#!/usr/bin/env python3
import torch
import torch_npu
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan, WanTransformer3DModel
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
import cache_dit
from cache_dit import (
	BlockAdapter,
	ForwardPattern,
	ParamsModifier,
	DBCacheConfig,
)

# 初始化npu环境
DEVICE_ID = 0                                         
torch.npu.set_device(DEVICE_ID)
device = f"npu:{DEVICE_ID}"

# 加载模型权重
model_path = "/data/weights/Wan2.2-T2V-A14B-Diffusers"  # 模型权重的保存路径
vae = AutoencoderKLWan.from_pretrained(
	model_path, subfolder="vae", torch_dtype=torch.float32
)
pipe = WanPipeline.from_pretrained(
	model_path, vae=vae, torch_dtype=torch.bfloat16
)

# 配置 scheduler（480P 使用 flow_shift=3.0，720P 使用 flow_shift=5.0）
height, width = 480, 832
flow_shift = 3.0 if height == 480 else 5.0
pipe.scheduler = UniPCMultistepScheduler.from_config(
	pipe.scheduler.config, flow_shift=flow_shift
)

# 由于模型较大（A14B MoE，总参数27B），需要使用 CPU offload
pipe.enable_model_cpu_offload()

# 启用 VAE tiling 和 slicing 以节省显存
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

# 确认是MoE双Transformer
assert isinstance(pipe.transformer, WanTransformer3DModel)
assert isinstance(pipe.transformer_2, WanTransformer3DModel)

# 使能Cache-DiT的DBCache功能（MoE双Transformer配置）
cache_dit.enable_cache(
	BlockAdapter(
		transformer=[
			pipe.transformer,
			pipe.transformer_2,
		],
		blocks=[
			pipe.transformer.blocks,
			pipe.transformer_2.blocks,
		],
		forward_pattern=[
			ForwardPattern.Pattern_2,
			ForwardPattern.Pattern_2,
		],
		params_modifiers=[
			ParamsModifier(
				cache_config=DBCacheConfig().reset(
					max_warmup_steps=4,
					max_cached_steps=8,
				),
			),
			ParamsModifier(
				cache_config=DBCacheConfig().reset(
					max_warmup_steps=2,
					max_cached_steps=20,
				),
			),
		],
		has_separate_cfg=True,
	),
	cache_config=DBCacheConfig(
		Fn_compute_blocks=8,
		Bn_compute_blocks=8,
		max_warmup_steps=8,
		max_cached_steps=-1,
		max_continuous_cached_steps=-1,
		residual_diff_threshold=0.10,
		enable_separate_cfg=True,
	),
)

# 定义推理步数分割函数（按boundary_ratio切分高低噪声步）
def split_inference_steps(num_inference_steps: int = 40) -> tuple[int, int]:
	if pipe.config.boundary_ratio is not None:
		boundary_timestep = pipe.config.boundary_ratio * pipe.scheduler.config.num_train_timesteps
	else:
		boundary_timestep = None
	pipe.scheduler.set_timesteps(num_inference_steps, device="npu")
	timesteps = pipe.scheduler.timesteps
	num_high_noise_steps = 0
	for t in timesteps:
		if boundary_timestep is not None and t >= boundary_timestep:
			num_high_noise_steps += 1
	num_low_noise_steps = num_inference_steps - num_high_noise_steps
	return num_high_noise_steps, num_low_noise_steps

# 在推理前对两个transformer分别刷新缓存上下文
num_inference_steps = 40
num_high_noise_steps, num_low_noise_steps = split_inference_steps(num_inference_steps)

cache_dit.refresh_context(
	pipe.transformer,
	num_inference_steps=num_high_noise_steps,
	verbose=True,
)
cache_dit.refresh_context(
	pipe.transformer_2,
	num_inference_steps=num_low_noise_steps,
	verbose=True,
)


# 准备输入
prompt = "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage."
negative_prompt = "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"

# 执行模型推理
with torch.inference_mode():
	output = pipe(
		prompt=prompt,
		negative_prompt=negative_prompt,
		height=height,
		width=width,
		num_frames=81,
		guidance_scale=4.0,
		guidance_scale_2=3.0,
		num_inference_steps=40,
	).frames[0]
	export_to_video(output, "t2v_out_cached.mp4", fps=16)
