import argparse
import numpy as np
import os
import torch
import torch_mlir

from import_utils import import_torch_module_with_fx
from models import bert_large, sd_clip_text_model, sd_unet_model, sd_vae_model, resnet50, t5_large, efficientnet_b7, efficientnet_v2_s

_MODEL_NAME_TO_MODEL_CONFIG = {
    #"BERT_LARGE": (bert_large.BertLarge, [1, 8]),
    #"RESNET50": (resnet50.Resnet50, [1, 8]),
    "SD_CLIP_TEXT_MODEL_SEQLEN64": (sd_clip_text_model.SDClipTextModel, [1]),
}


def import_to_mlir(model, batch_size, use_fp16):
    inputs = model.generate_inputs(batch_size=batch_size)
    if use_fp16:
        #inputs.to(device=torch.device("cuda"), dtype=torch.half)
        inputs = tuple([x.to(device=torch.device("cuda:0")) for x in inputs])
        #test_input.
    mlir_data = import_torch_module_with_fx(
        model, inputs, torch_mlir.OutputType.LINALG_ON_TENSORS, use_fp16)
    return (inputs, mlir_data)


if __name__ == "__main__":
    argParser = argparse.ArgumentParser()
    argParser.add_argument("-o",
                           "--output_dir",
                           default="/tmp",
                           help="Path to save mlir files")
    args = argParser.parse_args()

    for model_name, model_config in _MODEL_NAME_TO_MODEL_CONFIG.items():
        print(f"\n\n--- {model_name} -------------------------------------")
        use_fp16 = True
        model_class = model_config[0]
        batch_sizes = model_config[1]

        model_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)

        for batch_size in batch_sizes:
            save_dir = os.path.join(model_dir, f"batch_{batch_size}")
            os.makedirs(save_dir, exist_ok=True)

            try:
                # Remove all gradient info from models and tensors since these models are inference only.
                with torch.no_grad():
                    model_dtype = torch.float16 if use_fp16 else torch.float32
                    model = model_class(model_dtype=model_dtype)

                    torch.set_default_tensor_type(torch.cuda.FloatTensor)
                    if use_fp16:
                        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = True
                    else:
                        torch.backends.cuda.matmul.allow_tf32 = True

                    model.model.to("cuda:0")

                    #if use_fp16:
                    #    inputs = inputs.to(device=torch.device("cuda"), dtype=torch.half)
                        #model = model.half()
                        #model.eval()
                        #model.to("cuda")

                    inputs, mlir_data = import_to_mlir(model, batch_size, use_fp16)

                    # Save inputs.
                    for idx, input in enumerate(inputs):
                        input_path = os.path.join(save_dir, f"input_{idx}.npy")
                        print(f"Saving input {idx} to {input_path}")
                        np.save(input_path, input)

                    # Save mlir.
                    save_path = os.path.join(save_dir, "linalg.mlir")
                    with open(save_path, "wb") as mlir_file:
                        mlir_file.write(mlir_data)
                        print(f"Saved {save_path}")

                    # Save output.
                    outputs = model.forward(*inputs)
                    output_path = os.path.join(save_dir, f"output_0.npy")
                    print(f"Saving output 0 to {output_path}")
                    np.save(output_path, outputs)

            except Exception as e:
                print(f"Failed to import model {model_name}. Exception: {e}")
