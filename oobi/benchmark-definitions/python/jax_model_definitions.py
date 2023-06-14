import itertools
import string

import data_types
import data_types_builder
import input_data_definitions
import jax_output_data_definitions
import unique_ids

PARENT_GCS_DIR = "https://storage.googleapis.com/iree-model-artifacts/jax/jax_models_0.4.10_1684396752"

# Meta models.
RESNET50_FP32_JAX = data_types.MetaModel(
    id=unique_ids.MODEL_RESNET50_FP32_JAX,
    name="RESNET50_FP32_JAX",
    tags=["fp32", "cnn", "resnet"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel",
    data_type=data_types.DataType.FP32,
)

BERT_LARGE_FP32_JAX = data_types.MetaModel(
    id=unique_ids.MODEL_BERT_LARGE_FP32_JAX,
    name="BERT_LARGE_FP32_JAX",
    tags=["fp32", "transformer-encoder", "bert"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel",
    data_type=data_types.DataType.FP32,
)

T5_LARGE_FP32_JAX = data_types.MetaModel(
    id=unique_ids.MODEL_T5_LARGE_FP32_JAX,
    name="T5_LARGE_FP32_JAX",
    tags=["fp32", "transformer-encoder", "transformer-decoder", "t5"],
    framework_type=data_types.ModelFrameworkType.JAX,
    source_info=
    "https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model",
    data_type=data_types.DataType.FP32,
)

# Constants and functions help build batch templates.
BATCH_ID = lambda model_id: string.Template(model_id + "-batch${batch_size}")
BATCH_NAME = lambda name: string.Template(name + "_BATCH${batch_size}")
BATCH_TAG = string.Template("batch-${batch_size}")

# Resnet50 models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/resnet#transformers.FlaxResNetModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/resnet50
RESNET50_FP32_JAX_3X224X224XF32_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_RESNET50_FP32_JAX_3X224X224XF32),
    name=BATCH_NAME("RESNET50_FP32_JAX_3X224X224XF32"),
    tags=[BATCH_TAG],
    meta_model=RESNET50_FP32_JAX,
    inputs=input_data_definitions.IMAGENET_APPLES_JAX_3X224X224XF32_BATCHES,
    outputs=jax_output_data_definitions.RESNET50_FP32_JAX_2048X7X7XF32_BATCHES,
    artifacts=[
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/RESNET50/batch_${batch_size}/stablehlo.mlirbc"),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/RESNET50/batch_${batch_size}/hlo/jit_forward.before_optimizations.txt"
            ),
        ),
    ])
RESNET50_FP32_JAX_3X224X224XF32_BATCHES = data_types_builder.build_batch_models(
    template=RESNET50_FP32_JAX_3X224X224XF32_BATCH_TEMPLATE,
    batch_sizes=[1, 8, 64, 128, 256, 2048])

# Bert-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/bert#transformers.FlaxBertModel.
# Batch sizes from MLPerf A100 Configs: https://github.com/mlcommons/inference_results_v2.1/tree/master/closed/NVIDIA/configs/bert
BERT_LARGE_FP32_JAX_384XI32_BATCH_TEMPALTE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_BERT_LARGE_FP32_JAX_384XI32),
    name=BATCH_NAME("BERT_LARGE_JAX_384XI32"),
    tags=[BATCH_TAG],
    meta_model=BERT_LARGE_FP32_JAX,
    inputs=input_data_definitions.BERT_LARGE_SEQLEN384_I32_BATCHES,
    outputs=jax_output_data_definitions.
    BERT_LARGE_FP32_JAX_384X1024XF32_BATCHES,
    artifacts=[
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/BERT_LARGE/batch_${batch_size}/stablehlo.mlirbc"),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/BERT_LARGE/batch_${batch_size}/hlo/jit_forward.before_optimizations.txt"
            ),
        ),
    ])
BERT_LARGE_FP32_JAX_384XI32_BATCHES = data_types_builder.build_batch_models(
    template=BERT_LARGE_FP32_JAX_384XI32_BATCH_TEMPALTE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512, 1024, 1280])

# T5-Large models.
# Model implementation from https://huggingface.co/docs/transformers/model_doc/t5#transformers.FlaxT5Model
# Bert-Large batch sizes used for T5-Large models.
T5_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE = data_types_builder.ModelTemplate(
    id=BATCH_ID(unique_ids.MODEL_T5_LARGE_FP32_JAX_512XI32),
    name=BATCH_NAME("T5_LARGE_FP32_JAX_512XI32"),
    tags=[BATCH_TAG],
    meta_model=T5_LARGE_FP32_JAX,
    inputs=input_data_definitions.T5_LARGE_SEQLEN512_I32_BATCHES,
    outputs=jax_output_data_definitions.T5_LARGE_FP32_JAX_512X1024XF32_BATCHES,
    artifacts=[
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.MLIR_STABLEHLO,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/T5_LARGE/batch_${batch_size}/stablehlo.mlirbc"),
        ),
        data_types_builder.ModelArtifactTemplate(
            artifact_type=data_types.ModelArtifactType.JAX_HLO_DUMP,
            source_url=string.Template(
                PARENT_GCS_DIR +
                "/T5_LARGE/batch_${batch_size}/hlo/jit_forward.before_optimizations.txt"
            ),
        ),
    ])
T5_LARGE_FP32_JAX_512XI32_BATCHES = data_types_builder.build_batch_models(
    template=T5_LARGE_FP32_JAX_512XI32_BATCH_TEMPLATE,
    batch_sizes=[1, 16, 24, 32, 48, 64, 512])

# Collections.
JAX_MODELS = list(
    itertools.chain(RESNET50_FP32_JAX_3X224X224XF32_BATCHES.values(),
                    BERT_LARGE_FP32_JAX_384XI32_BATCHES.values(),
                    T5_LARGE_FP32_JAX_512XI32_BATCHES.values()))
JAX_MODELS_DICT = dict((model.id, model) for model in JAX_MODELS)
