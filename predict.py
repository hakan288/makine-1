from cog import BasePredictor, Input, Path
import torch
from diffusers import (
    StableDiffusionXLControlNetPipeline,
    ControlNetModel,
)

from PIL import Image

class Predictor(BasePredictor):
    def setup(self):
        # ControlNet’leri OpenPose ve Depth gibi yükle
        self.controlnet_openpose = ControlNetModel.from_pretrained(
            "diffusers/controlnet-openpose-sdxl-1.0", torch_dtype=torch.float16
        )
        self.controlnet_depth = ControlNetModel.from_pretrained(
            "diffusers/controlnet-depth-sdxl-1.0", torch_dtype=torch.float16
        )

        # SDXL temel model + multi ControlNet
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            controlnet=[self.controlnet_openpose, self.controlnet_depth],
            torch_dtype=torch.float16
        ).to("cuda")

    def predict(
        self,
        prompt: str = Input(description="Scene prompt"),
        negative_prompt: str = Input(description="Negative prompt"),
        reference_image: Path = Input(description="Cover/reference image"),
        openpose_image: Path = Input(description="OpenPose skeleton image"),
        depth_image: Path = Input(description="Optional Depth image"),
        num_inference_steps: int = Input(default=30),
        guidance_scale: float = Input(default=7.5),
        seed: int = Input(default=None),
    ) -> Path:

        ref_img = Image.open(reference_image).convert("RGB")
        pose_img = Image.open(openpose_image).convert("RGB")
        depth_img = Image.open(depth_image).convert("RGB") if depth_image else None

        generator = None
        if seed:
            generator = torch.manual_seed(seed)

        output = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=ref_img,
            control_image=[pose_img, depth_img] if depth_img else [pose_img],
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        ).images[0]

        output_path = "/tmp/output.png"
        output.save(output_path)
        return Path(output_path)
