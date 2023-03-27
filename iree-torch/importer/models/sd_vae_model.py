import torch

from diffusers import AutoencoderKL
from models.input_data import imagenet_test_data
from torchvision import transforms


# `VAE`: Variational auto-encoder.
# Compresses an input image into latent space using its encoder.
# Uncompresses latents into images using the decoder.
# Allows Stable Diffusion to perform diffusion in the latent space and convert
# to a higher resolution image using the `VAE` decoder.
class SDVaeModel(torch.nn.Module):

    def __init__(self, model_dtype=torch.float16):
        super().__init__()
        self.model_dtype = model_dtype
        self.model = AutoencoderKL.from_pretrained(
            "CompVis/stable-diffusion-v1-4", subfolder="vae")

    def generate_inputs(self, batch_size=1):
        # We use VAE to encode an image and return this for input to the VAE decoder.
        image = imagenet_test_data.get_image_input().resize([512, 512])
        input = transforms.ToTensor()(image).unsqueeze(0)
        input = input.repeat(batch_size, 1, 1, 1)
        output = self.model.encode(input)
        sample = output.latent_dist.sample()
        return (sample, )

    def forward(self, input):
        return self.model.decode(input, return_dict=False)[0]
