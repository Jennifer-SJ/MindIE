#!/usr/bin/env python
# coding=utf-8
# Copyright (c) Huawei Technologies Co., Ltd. 2026-2026. All rights reserved.
# MindIE is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#          http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

"""Wan2.2 T2V 单卡验证脚本。

这个脚本有两个互斥的运行模式：

1. baseline：只使用 diffusers 原生 WanPipeline；
2. compile：在同一条 diffusers 推理链路上，为两个 Wan Transformer 使能
   MindieSDBackend 区域编译。

脚本暂时不接入 Cache-DiT，目的是控制变量，只判断 MindIE-SD
compile 是否生效。
Wan2.2 A14B 同时包含高噪声和低噪声两个 Transformer，单卡无法让所有权重长期
驻留在 NPU，因此两个模式都使用 model CPU offload。
"""

import argparse
import logging
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch_npu
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video


DEFAULT_PROMPT = (
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a "
    "spotlighted stage."
)
DEFAULT_NEGATIVE_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，"
    "画作，画面，"
    "静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，"
    "多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，"
    "形态畸形的肢体，手指融合，"
    "静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)
GIB = 1024**3
COMPILE_PATTERN_NAMES = (
    "rms_norm",
    "rope",
    "adalayernorm",
    "fast_gelu",
    "mul_add",
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """解析命令行参数，并提前拦截明显不合法的视频规格。"""
    parser = argparse.ArgumentParser(
        description="Validate diffusers Wan2.2 T2V baseline and MindIE-SD compile on one NPU."
    )
    parser.add_argument(
        "--model-path",
        default="/data/weights/Wan2.2-T2V-A14B-Diffusers",
        help="Local diffusers-format Wan2.2 model directory.",
    )
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument(
        "--warmup-inference-steps",
        type=int,
        default=8,
        help="Diffusion steps for one warmup run outside timing/profiling; set 0 to disable.",
    )
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--guidance-scale-2", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument(
        "--output-type",
        choices=("latent", "video"),
        default="video",
        help="Use latent for DiT-only measurement or video for end-to-end quality validation.",
    )
    parser.add_argument("--output", default=None, help="Output .pt or .mp4 path.")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile WanTransformerBlock modules with MindieSDBackend.",
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="Require each repeated block to be captured as one full graph.",
    )
    parser.add_argument(
        "--disable-compile-pattern",
        action="append",
        choices=("all", *COMPILE_PATTERN_NAMES),
        default=[],
        help=(
            "Disable one MindIE-SD fusion pattern for diagnosis; repeat this option to disable "
            "multiple patterns, or use 'all' to disable every fusion pattern."
        ),
    )
    parser.add_argument("--profile", action="store_true", help="Collect an NPU profiler trace.")
    parser.add_argument(
        "--profile-dir",
        default=None,
        help="Profiler output directory; defaults to profile_wan22_baseline/compile.",
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError("height and width must be divisible by 16")
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError("num_frames must satisfy (num_frames - 1) % 4 == 0")
    if args.num_inference_steps <= 0:
        raise ValueError("num_inference_steps must be greater than 0")
    if args.warmup_inference_steps < 0:
        raise ValueError("warmup_inference_steps must not be negative")
    return args


def enable_mindie_compile(
    pipe: WanPipeline,
    fullgraph: bool,
    disabled_patterns: list[str],
) -> None:
    """为 Wan2.2 的两个 Transformer Block 使能 MindIE-SD 区域编译。

    ``compile_repeated_blocks`` 不编译整个 Pipeline，只编译模型里反复出现的
    ``WanTransformerBlock``。真正的图捕获和融合算子替换发生在第一次 forward，
    所以后面的 warmup 是 compile 验证中不可省略的一步。
    """
    from mindiesd.compilation import MindieSDBackend
    from mindiesd.compilation.compiliation_config import CompilationConfig

    patterns_to_disable = set(disabled_patterns)
    if "all" in patterns_to_disable:
        patterns_to_disable = set(COMPILE_PATTERN_NAMES)
    for pattern_name in patterns_to_disable:
        config_name = "enable_%s" % pattern_name
        setattr(CompilationConfig.fusion_patterns, config_name, False)
    logger.info(
        "Disabled MindIE-SD fusion patterns: %s",
        ", ".join(sorted(patterns_to_disable)) if patterns_to_disable else "none",
    )

    backend = MindieSDBackend()
    # Wan2.2 在高噪声阶段使用 transformer，在低噪声阶段使用 transformer_2。
    # 两个模型都要进入 MindieSDBackend，否则只能加速其中一部分去噪步骤。
    for module_name in ("transformer", "transformer_2"):
        transformer = getattr(pipe, module_name, None)
        if transformer is None:
            raise RuntimeError("Wan2.2 requires %s, but it is missing" % module_name)
        if not hasattr(transformer, "compile_repeated_blocks"):
            raise RuntimeError(
                "%s does not provide compile_repeated_blocks; install a recent diffusers version"
                % module_name
            )
        transformer.compile_repeated_blocks(
            backend=backend,
            fullgraph=fullgraph,
            # 当前验证的分辨率、帧数和 batch size 固定。
            # 关闭动态 shape 可以减少
            # guard 和重复编译干扰，让 baseline/compile 对比更容易解释。
            dynamic=False,
        )
        logger.info("Enabled MindieSDBackend regional compile for %s", module_name)


def load_pipeline(args: argparse.Namespace, device: str) -> WanPipeline:
    """从本地 diffusers 权重目录加载并配置 Wan2.2 Pipeline。"""
    logger.info("Loading Wan2.2 from %s", args.model_path)

    # Wan VAE 对数值精度更敏感，因此保持 FP32。
    # 两个 Transformer 由 Pipeline 以 BF16 加载。
    vae = AutoencoderKLWan.from_pretrained(
        args.model_path,
        subfolder="vae",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    pipe = WanPipeline.from_pretrained(
        args.model_path,
        vae=vae,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    )

    # UniPC 的 flow_shift 与输出分辨率相关：Wan 官方配置中 480P 使用 3.0，
    # 720P 使用 5.0。本脚本沿用基线脚本的规则。
    flow_shift = 3.0 if args.height == 480 else 5.0
    pipe.scheduler = UniPCMultistepScheduler.from_config(
        pipe.scheduler.config,
        flow_shift=flow_shift,
    )
    pipe.vae.enable_tiling()
    pipe.vae.enable_slicing()

    # 先标记需要编译的 Block，再安装 CPU offload hook。这样编译只关注 Block 的
    # 张量计算，不需要把 Accelerate 的模型搬运逻辑一起捕获到计算图中。
    if args.compile:
        enable_mindie_compile(pipe, args.fullgraph, args.disable_compile_pattern)

    # model CPU offload 以“完整组件”为粒度搬运模型。
    # 当前需要执行哪个组件时，Accelerate 将它从 CPU 搬到 NPU；
    # 下一个组件执行前再把前一个组件移回 CPU。
    pipe.enable_model_cpu_offload(device=device)
    return pipe


def run_pipeline(
    pipe: WanPipeline,
    args: argparse.Namespace,
    inference_steps: int,
    output_type: str,
):
    """执行一次 Pipeline；调用方决定扩散步数和是否跳过 VAE。"""
    # 每次调用都从相同 seed 创建新 Generator，
    # 确保 warmup 不会改变正式运行的噪声。
    generator = torch.Generator("cpu").manual_seed(args.seed)

    # latent 模式在去噪完成后直接返回 latent，不执行 VAE，
    # 适合定位 DiT 性能；
    # video 模式返回帧数组，适合检查完整链路和生成质量。
    diffusers_output_type = "latent" if output_type == "latent" else "np"
    return pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        num_inference_steps=inference_steps,
        generator=generator,
        output_type=diffusers_output_type,
    ).frames[0]


def create_profiler(args: argparse.Namespace):
    """按需创建 NPU Profiler；未指定 --profile 时返回空上下文。"""
    if not args.profile:
        return nullcontext()

    default_dir = "profile_wan22_compile" if args.compile else "profile_wan22_baseline"
    profile_dir = Path(args.profile_dir or default_dir).resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Profiler output: %s", profile_dir)
    return torch_npu.profiler.profile(
        activities=[torch_npu.profiler.ProfilerActivity.NPU],
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
        with_stack=True,
        record_shapes=True,
        profile_memory=True,
    )


def save_output(output, args: argparse.Namespace) -> Path:
    """保存 latent 张量或导出 MP4，保存时间不计入推理耗时。"""
    if args.output is None:
        mode = "compile" if args.compile else "baseline"
        suffix = "pt" if args.output_type == "latent" else "mp4"
        output_path = Path("t2v_%s.%s" % (mode, suffix))
    else:
        output_path = Path(args.output)
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.output_type == "latent":
        torch.save(output.detach().float().cpu(), output_path)
    else:
        export_to_video(output, str(output_path), fps=16)
    return output_path


def main() -> None:
    """完成环境检查、模型加载、warmup、正式推理和结果保存。"""
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
    )

    if not Path(args.model_path).is_dir():
        raise FileNotFoundError("Model directory does not exist: %s" % args.model_path)
    if not torch.npu.is_available():
        raise RuntimeError("No available Ascend NPU was detected")

    # 进程必须先绑定 NPU，后续 offload 才知道把模型搬到哪张卡。
    torch.npu.set_device(args.device_id)
    device = "npu:%d" % args.device_id
    mode = "compile" if args.compile else "baseline"
    logger.info("Mode: %s", mode)
    logger.info("Device: %s", device)
    logger.info(
        "Workload: %dx%d, frames=%d, steps=%d, output_type=%s",
        args.height,
        args.width,
        args.num_frames,
        args.num_inference_steps,
        args.output_type,
    )
    if args.profile:
        logger.warning(
            "Profiler overhead is included in elapsed_seconds; use a non-profile run for timing"
        )

    # 模型加载发生在 CPU。只有执行 forward 时，
    # CPU offload 才按需把组件搬到 NPU。
    pipe = load_pipeline(args, device)
    logger.info("boundary_ratio=%s", pipe.config.boundary_ratio)
    logger.info("transformer blocks=%d", len(pipe.transformer.blocks))
    logger.info("transformer_2 blocks=%d", len(pipe.transformer_2.blocks))

    if args.warmup_inference_steps > 0:
        logger.info("Warmup with %d diffusion steps", args.warmup_inference_steps)
        # Warmup 使用相同 shape，但固定返回 latent，
        # 避免把 VAE 解码混入编译准备阶段。
        # compile 模式下，首次执行到两个 Transformer 的 Block 时
        # 才真正触发图编译。
        with torch.inference_mode():
            run_pipeline(pipe, args, args.warmup_inference_steps, "latent")
        torch.npu.synchronize()
        # empty_cache 只释放 PyTorch 缓存池中当前没有使用的显存，
        # 不会删除模型权重。
        torch.npu.empty_cache()

    # reset_peak_memory_stats 只清零显存峰值计数器，便于报告正式运行的峰值。
    torch.npu.reset_peak_memory_stats(args.device_id)

    # NPU 默认异步执行。计时前后都 synchronize，
    # 确保 elapsed_seconds 覆盖真实计算。
    torch.npu.synchronize()
    start_time = time.perf_counter()

    # Profiler 放在 warmup 之后，只采正式运行。
    # 未开启 --profile 时这是一个空上下文。
    with create_profiler(args):
        with torch.inference_mode():
            output = run_pipeline(
                pipe,
                args,
                args.num_inference_steps,
                args.output_type,
            )
        torch.npu.synchronize()
    elapsed_seconds = time.perf_counter() - start_time
    peak_memory_gib = torch.npu.max_memory_allocated(args.device_id) / GIB

    output_path = save_output(output, args)
    output_shape = tuple(output.shape) if hasattr(output, "shape") else "video frames"
    logger.info("Output shape: %s", output_shape)
    logger.info("Output saved to: %s", output_path)
    logger.info(
        "RESULT mode=%s elapsed_seconds=%.3f peak_memory_gib=%.3f",
        mode,
        elapsed_seconds,
        peak_memory_gib,
    )


if __name__ == "__main__":
    main()
