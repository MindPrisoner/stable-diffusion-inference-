import os
import torch
from diffusers import StableDiffusionPipeline


def generate_image(
    model_path,
    prompt,
    negative_prompt,
    output_path,
    steps=20,
    guidance=7.5,
    height=384,
    width=384,
):
    use_cuda = torch.cuda.is_available()
    dtype = torch.float16 if use_cuda else torch.float32

    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
        local_files_only=True,
    )

    if use_cuda:
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
    else:
        pipe = pipe.to("cpu")

    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=height,
        width=width,
    ).images[0]

    image.save(output_path)
    print(f"Image saved to {output_path}")


def main():
    os.makedirs("outputs", exist_ok=True)

    model_path = "/mnt/d/AIModels/sd15"
    prompt = "a cute corgi wearing sunglasses, highly detailed, studio lighting"
    negative_prompt = "blurry, low quality, distorted, ugly"
    generate_image(
        model_path=model_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        output_path="outputs/have_negative.png",
        steps=20,
        guidance=7.5,
        height=384,
        width=384,
    )


if __name__ == "__main__":
    main()