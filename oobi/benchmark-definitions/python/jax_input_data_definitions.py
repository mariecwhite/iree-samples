import string

import data_types
import data_types_builder
import unique_ids

# Constants and functions help build batch templates.
BATCH_ID = lambda data_id: string.Template(data_id + "-batch${batch_size}")
BATCH_NAME = lambda name: string.Template(name + "_BATCH${batch_size}")
BATCH_TAG = string.Template("batch-${batch_size}")
BATCH_MODEL_ID = lambda model_id: string.Template(model_id +
                                                  "-batch${batch_size}")
BATCH_TENSOR_DIMS = lambda dims: string.Template("${batch_size}x" + dims)


IMAGENET_APPLES_JAX_3X224X224XF32_BATCH_TEMPLATE = data_types_builder.ModelDataTemplate(
    id=BATCH_ID(unique_ids.INPUT_DATA_IMAGENET_APPLES_JAX_3X224X224XF32),
    name=BATCH_NAME("IMAGENET_APPLES_JAX_3X224X224XF32"),
    tags=["input-data", "imagenet", BATCH_TAG],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=BATCH_MODEL_ID(unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32),
    source_info=
    "Original image: https://storage.googleapis.com/iree-model-artifacts/ILSVRC2012_val_00000023.JPEG",
    tensor_names=["pixel_values"],
    tensor_dimensions=[BATCH_TENSOR_DIMS(dims="3x224x224xf32")],
    source_url=[
        string.Template(
            "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752/RESNET50/batch_${batch_size}/input_0.npy"
        )
    ])
IMAGENET_APPLES_JAX_3X224X224XF32_BATCHES = data_types_builder.build_batch_model_data(
    template=IMAGENET_APPLES_JAX_3X224X224XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])


BERT_LARGE_JAX_SEQLEN384_I32_BATCH_TEMPLATE = data_types_builder.ModelDataTemplate(
    id=BATCH_ID(unique_ids.INPUT_DATA_BERT_LARGE_JAX_SEQLEN384_I32),
    name=BATCH_NAME("BERT_LARGE_JAX_SEQLEN384_I32"),
    tags=["input-data", "seqlen384", BATCH_TAG],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=BATCH_MODEL_ID(unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32),
    source_info="Original text: 'a photo of a cat'.",
    tensor_names=["input_ids", "attention_mask"],
    tensor_dimensions=[
        BATCH_TENSOR_DIMS("384xi32"),
        BATCH_TENSOR_DIMS("384xi32")
    ],
    source_url=[
        string.Template(
            "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752/BERT_LARGE/batch_${batch_size}/input_0.npy"
        ),
        string.Template(
            "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752/BERT_LARGE/batch_${batch_size}/input_1.npy"
        ),
    ])
BERT_LARGE_JAX_SEQLEN384_I32_BATCHES = data_types_builder.build_batch_model_data(
    template=BERT_LARGE_JAX_SEQLEN384_I32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])


T5_LARGE_JAX_SEQLEN512_I32_BATCH_TEMPLATE = data_types_builder.ModelDataTemplate(
    id=BATCH_ID(unique_ids.INPUT_DATA_T5_LARGE_JAX_SEQLEN512_I32),
    name=BATCH_NAME("T5_LARGE_JAX_SEQLEN512_I32"),
    tags=["input-data", "seqlen512", BATCH_TAG],
    data_format=data_types.DataFormat.NUMPY_NPY,
    model_id=BATCH_MODEL_ID(unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32),
    source_info=
    "Original text: 'Studies have been shown that owning a dog is good for you'",
    tensor_names=["input_ids", "decoder_input_ids"],
    tensor_dimensions=[
        BATCH_TENSOR_DIMS("512xi32"),
        BATCH_TENSOR_DIMS("512xi32")
    ],
    source_url=[
        string.Template(
            "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752/T5_LARGE/batch_${batch_size}/input_0.npy"
        ),
        string.Template(
            "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752/T5_LARGE/batch_${batch_size}/input_1.npy"
        ),
    ],
)
T5_LARGE_JAX_SEQLEN512_I32_BATCHES = data_types_builder.build_batch_model_data(
    template=T5_LARGE_JAX_SEQLEN512_I32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])
