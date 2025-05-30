#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import logging
import sys
from pathlib import Path

import torch
from diffusers.utils import export_to_gif, export_to_video, load_image

from optimum.habana.diffusers import (
    GaudiEulerDiscreteScheduler,
    GaudiI2VGenXLPipeline,
    GaudiStableVideoDiffusionPipeline,
)
from optimum.habana.utils import set_seed


try:
    from optimum.habana.utils import check_optimum_habana_min_version
except ImportError:

    def check_optimum_habana_min_version(*a, **b):
        return ()


# Will error if the minimal version of Optimum Habana is not installed. Remove at your own risks.
check_optimum_habana_min_version("1.18.0.dev0")


logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name_or_path",
        default="stabilityai/stable-video-diffusion-img2vid-xt",
        type=str,
        help="Path to pre-trained model",
    )
    parser.add_argument(
        "--controlnet_model_name_or_path",
        default="CiaraRowles/temporal-controlnet-depth-svd-v1",
        type=str,
        help="Path to pre-trained controlnet model.",
    )

    # Pipeline arguments
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="*",
        default="Papers were floating in the air on a table in the library",
        help="The prompt or prompts to guide the image generation.",
    )
    parser.add_argument(
        "--negative_prompts",
        type=str,
        nargs="*",
        default="Distorted, discontinuous, Ugly, blurry, low resolution, motionless, static, disfigured, disconnected limbs, Ugly faces, incomplete arms",
        help="The prompt or prompts not to guide the image generation.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        nargs="*",
        help="Path to input image(s) to guide video generation",
    )
    parser.add_argument(
        "--control_image_path",
        type=str,
        default=None,
        nargs="*",
        help="Path to controlnet input image(s) to guide video generation.",
    )
    parser.add_argument(
        "--num_videos_per_prompt", type=int, default=1, help="The number of videos to generate per prompt image."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="The number of videos in a batch.")
    parser.add_argument("--height", type=int, default=576, help="The height in pixels of the generated video.")
    parser.add_argument("--width", type=int, default=1024, help="The width in pixels of the generated video.")
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help=(
            "The number of denoising steps. More denoising steps usually lead to a higher quality images at the expense"
            " of slower inference."
        ),
    )
    parser.add_argument(
        "--min_guidance_scale",
        type=float,
        default=1.0,
        help="The minimum guidance scale. Used for the classifier free guidance with first frame.",
    )
    parser.add_argument(
        "--max_guidance_scale",
        type=float,
        default=3.0,
        help="The maximum guidance scale. Used for the classifier free guidance with last frame.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=7,
        help=(
            "Frames per second. The rate at which the generated images shall be exported to a video after generation."
            " Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training."
        ),
    )
    parser.add_argument(
        "--motion_bucket_id",
        type=int,
        default=127,
        help=(
            "The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion"
            " will be in the video."
        ),
    )
    parser.add_argument(
        "--noise_aug_strength",
        type=float,
        default=0.02,
        help=(
            "The amount of noise added to the init image, the higher it is the less the video will look like the"
            " init image. Increase it for more motion."
        ),
    )
    parser.add_argument(
        "--decode_chunk_size",
        type=int,
        default=None,
        help=(
            "The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency"
            " between frames, but also the higher the memory consumption. By default, the decoder will decode all"
            " frames at once for maximal quality. Reduce `decode_chunk_size` to reduce memory usage."
        ),
    )
    parser.add_argument(
        "--output_type",
        type=str,
        choices=["pil", "np"],
        default="pil",
        help="Whether to return PIL images or Numpy arrays.",
    )
    parser.add_argument(
        "--pipeline_save_dir",
        type=str,
        default=None,
        help="The directory where the generation pipeline will be saved.",
    )
    parser.add_argument(
        "--video_save_dir",
        type=str,
        default="./stable-video-diffusion-generated-frames",
        help="The directory where frames will be saved.",
    )
    parser.add_argument(
        "--save_frames_as_images",
        action="store_true",
        help="Save output frames as images",
    )

    parser.add_argument("--seed", type=int, default=42, help="Random seed for initialization.")

    # HPU-specific arguments
    parser.add_argument("--use_habana", action="store_true", help="Use HPU.")
    parser.add_argument(
        "--use_hpu_graphs", action="store_true", help="Use HPU graphs on HPU. This should lead to faster generations."
    )
    parser.add_argument(
        "--gaudi_config_name",
        type=str,
        default="Habana/stable-diffusion",
        help=(
            "Name or path of the Gaudi configuration. In particular, it enables to specify how to apply Habana Mixed"
            " Precision."
        ),
    )
    parser.add_argument("--bf16", action="store_true", help="Whether to perform generation in bf16 precision.")
    parser.add_argument("--gif", action="store_true", help="Whether to generate the video in gif format.")
    parser.add_argument(
        "--sdp_on_bf16",
        action="store_true",
        default=False,
        help="Allow pyTorch to use reduced precision in the SDPA math backend",
    )
    parser.add_argument("--num_frames", type=int, default=25, help="The number of video frames to generate.")
    parser.add_argument(
        "--profiling_warmup_steps",
        default=0,
        type=int,
        help="Number of steps to ignore for profiling.",
    )
    parser.add_argument(
        "--profiling_steps",
        default=0,
        type=int,
        help="Number of steps to capture for profiling.",
    )
    parser.add_argument(
        "--throughput_warmup_steps",
        type=int,
        default=None,
        help="Number of steps to ignore for throughput calculation.",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)

    i2v_models = ["i2vgen-xl"]
    is_i2v_model = any(model in args.model_name_or_path for model in i2v_models)

    # Load input image(s)
    input = []
    logger.info("Input image(s):")
    if isinstance(args.image_path, str):
        args.image_path = [args.image_path]
    for image_path in args.image_path:
        image = load_image(image_path)
        if is_i2v_model:
            image = image.convert("RGB")
        else:
            image = image.resize((args.height, args.width))
        input.append(image)
        logger.info(image_path)

    # Load control input image
    control_input = []
    if args.control_image_path is not None:
        logger.info("Input control image(s):")
        if isinstance(args.control_image_path, str):
            args.control_image_path = [args.control_image_path]
        for control_image in args.control_image_path:
            image = load_image(control_image)
            image = image.resize((args.height, args.width))
            control_input.append(image)
            logger.info(control_image)

    # Initialize the scheduler and the generation pipeline
    scheduler = GaudiEulerDiscreteScheduler.from_pretrained(args.model_name_or_path, subfolder="scheduler")
    kwargs = {
        "scheduler": scheduler,
        "use_habana": args.use_habana,
        "use_hpu_graphs": args.use_hpu_graphs,
        "gaudi_config": args.gaudi_config_name,
        "sdp_on_bf16": args.sdp_on_bf16,
    }

    set_seed(args.seed)
    if args.bf16:
        kwargs["torch_dtype"] = torch.bfloat16

    if args.control_image_path is not None:
        from optimum.habana.diffusers import GaudiStableVideoDiffusionControlNetPipeline
        from optimum.habana.diffusers.models import ControlNetSDVModel, UNetSpatioTemporalConditionControlNetModel

        controlnet = ControlNetSDVModel.from_pretrained(
            args.controlnet_model_name_or_path, subfolder="controlnet", **kwargs
        )
        unet = UNetSpatioTemporalConditionControlNetModel.from_pretrained(
            args.model_name_or_path, subfolder="unet", **kwargs
        )
        pipeline = GaudiStableVideoDiffusionControlNetPipeline.from_pretrained(
            args.model_name_or_path, controlnet=controlnet, unet=unet, **kwargs
        )

        # Generate images
        outputs = pipeline(
            image=input,
            controlnet_condition=control_input,
            num_videos_per_prompt=args.num_videos_per_prompt,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            min_guidance_scale=args.min_guidance_scale,
            max_guidance_scale=args.max_guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            noise_aug_strength=args.noise_aug_strength,
            decode_chunk_size=args.decode_chunk_size,
            output_type=args.output_type,
            num_frames=args.num_frames,
        )
    elif is_i2v_model:
        del kwargs["scheduler"]
        pipeline = GaudiI2VGenXLPipeline.from_pretrained(
            args.model_name_or_path,
            **kwargs,
        )
        generator = torch.manual_seed(args.seed)
        outputs = pipeline(
            prompt=args.prompts,
            image=input,
            num_videos_per_prompt=args.num_videos_per_prompt,
            batch_size=args.batch_size,
            num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps,
            negative_prompt=args.negative_prompts,
            guidance_scale=9.0,
            generator=generator,
        )
    else:
        pipeline = GaudiStableVideoDiffusionPipeline.from_pretrained(
            args.model_name_or_path,
            **kwargs,
        )
        kwargs_call = {}
        if args.throughput_warmup_steps is not None:
            kwargs_call["throughput_warmup_steps"] = args.throughput_warmup_steps

        # Generate images
        outputs = pipeline(
            image=input,
            num_videos_per_prompt=args.num_videos_per_prompt,
            batch_size=args.batch_size,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            min_guidance_scale=args.min_guidance_scale,
            max_guidance_scale=args.max_guidance_scale,
            fps=args.fps,
            motion_bucket_id=args.motion_bucket_id,
            noise_aug_strength=args.noise_aug_strength,
            decode_chunk_size=args.decode_chunk_size,
            output_type=args.output_type,
            profiling_warmup_steps=args.profiling_warmup_steps,
            profiling_steps=args.profiling_steps,
            **kwargs_call,
        )

    # Save the pipeline in the specified directory if not None
    if args.pipeline_save_dir is not None:
        pipeline.save_pretrained(args.pipeline_save_dir)

    # Save images in the specified directory if not None and if they are in PIL format
    if args.video_save_dir is not None:
        if args.output_type == "pil":
            video_save_dir = Path(args.video_save_dir)
            video_save_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Saving video frames in {video_save_dir.resolve()}...")
            for i, frames in enumerate(outputs.frames):
                if args.gif:
                    export_to_gif(frames, args.video_save_dir + "/gen_video_" + str(i).zfill(2) + ".gif")
                else:
                    export_to_video(frames, args.video_save_dir + "/gen_video_" + str(i).zfill(2) + ".mp4", fps=7)

                if args.save_frames_as_images:
                    for j, frame in enumerate(frames):
                        frame.save(
                            args.video_save_dir
                            + "/gen_video_"
                            + str(i).zfill(2)
                            + "_frame_"
                            + str(j).zfill(2)
                            + ".png"
                        )
        else:
            logger.warning("--output_type should be equal to 'pil' to save frames in --video_save_dir.")


if __name__ == "__main__":
    main()
