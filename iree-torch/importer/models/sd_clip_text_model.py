import torch
from transformers import AutoTokenizer, CLIPTextModel

_SEQUENCE_LENGTH = 77


# `CLIP``: Contrastive Language–Image Pre-training.
# Takes text input and outputs text embeddings.
# The output of the ClipTextModel becomes one of the inputs to the `UNet` in a
# Stable Diffusion pipeline.
class SDClipTextModel(torch.nn.Module):

    def __init__(self, model_dtype=torch.float32):
        super().__init__()
        self.model_dtype = model_dtype
        if self.model_dtype == torch.float16:
            self.model = CLIPTextModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="text_encoder",
                revision="fp16",
                torch_dtype=torch.float16,
            )
        else:
            self.model = CLIPTextModel.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                subfolder="text_encoder",
            )

    def generate_inputs(self, batch_size=1):
        tokenizer = AutoTokenizer.from_pretrained(
            "openai/clip-vit-base-patch32")
        input_text = ["a photo of a cat"] * batch_size
        inputs = tokenizer(text=input_text,
                           padding="max_length",
                           max_length=_SEQUENCE_LENGTH,
                           return_tensors="pt")
        return (inputs["input_ids"], inputs["attention_mask"])

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids, attention_mask)[0]
