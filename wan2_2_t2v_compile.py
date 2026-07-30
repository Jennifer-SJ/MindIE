# 使能compile能力

#!/usr/bin/env python3
import torch
import torch_npu
import numpy as np
from diffusers import WanPipeline, AutoencoderKLWan
from diffusers.utils import export_to_video
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from mindiesd.compilation import MindieSDBackend


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

# 使能MindIE-SD的编译优化（对两个transformer分别编译）
import torch._dynamo
# 调大重新编译的次数，保证从eager到图模式能生效
torch._dynamo.config.recompile_limit = 1024 # default 8
torch._dynamo.config.accumulated_recompile_limit = 8129

pipe.transformer = torch.compile(
	pipe.transformer,
	backend=MindieSDBackend(),
)
pipe.transformer_2 = torch.compile(
	pipe.transformer_2,
	backend=MindieSDBackend(),
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
		generator=torch.Generator("cpu").manual_seed(0)
	).frames[0]
	export_to_video(output, "t2v_out_compile.mp4", fps=16)
