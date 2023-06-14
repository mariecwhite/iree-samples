import argparse
import jax
import jax.numpy as jnp
import json
import multiprocessing
import numpy as np
import os
import pathlib
import requests
import statistics
import sys
import time

from typing import Optional, Any


# Add library dir to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent / "library"))
from models import bert_large, resnet50, t5_large

# Add benchmark definitions to the search path.
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "oobi" / "benchmark-definitions" / "python"))
import data_types, jax_model_definitions, unique_ids
from utils import execution_environment

def benchmark_lookup(unique_id: str):
  if unique_id not in jax_model_definitions.JAX_MODELS_DICT:
    id_list = '\n  '.join(jax_model_definitions.JAX_MODELS_DICT.keys())
    raise ValueError(f"Id {unique_id} does not exist in model suite. Expected "
                     f"one of:\n  {id_list}")

  model_definition = jax_model_definitions.JAX_MODELS_DICT[unique_id]
  if unique_id.startswith(unique_ids.MODEL_RESNET50_FP32_JAX):
    return ("RESNET50", resnet50.ResNet50, model_definition)
  elif unique_id.startswith(unique_ids.MODEL_BERT_LARGE_FP32_JAX):
    return ("BERT_LARGE", bert_large.BertLarge, model_definition)
  elif unique_id.startswith(unique_ids.MODEL_T5_LARGE_FP32_JAX):
    return ("T5_LARGE", t5_large.T5Large, model_definition)
  else:
    raise ValueError(f"Model definition not supported")


def dump_result(file_path: str, result: dict) -> None:
  with open(file_path, "r") as f:
    dictObj = json.load(f)

  dictObj["execution_environment"] = {
      "python_environment": execution_environment.get_python_environment_info()
  }
  dictObj["benchmarks"].append(result)

  with open(file_path, "w") as f:
    json.dump(dictObj, f)


def bytes_to_mb_str(bytes: Optional[int]) -> str:
  return "n/a" if bytes is None else f"{bytes / 1e6:.6f}"

def retrieve_data(model_data: data_types.ModelData, cache_dir: str) -> tuple[Any, ...]:
  data = ()
  for url in model_data.source_url:
    assert url.startswith("https://storage.googleapis.com/iree-model-artifacts/")
    relative_path = url.removeprefix("https://storage.googleapis.com/iree-model-artifacts/")
    local_path = cache_dir / relative_path

    if not os.path.exists(local_path):
      pathlib.Path(os.path.dirname(local_path)).mkdir(parents=True, exist_ok=True)
      r = requests.get(url)
      print(f"Downloading {url}")
      open(local_path, 'wb').write(r.content)

    array = np.load(local_path)
    data = data + (array, )
  return data

def run_framework_benchmark(model_name: str, model_class: Any,
                            input_data: tuple[np.array, ...],
                            expected_outputs: tuple[np.array, ...],
                            warmup_iterations: int, benchmark_iterations: int,
                            backend: str, shared_dict) -> None:
  try:
    with jax.default_device(jax.devices(backend)[0]):
      model = model_class()

      # Create jits.
      start = time.perf_counter()
      jit_inputs = jax.device_put(input_data)
      end = time.perf_counter()
      input_data_transfer_ms = 1000 * (end - start)

      jit_function = jax.jit(model.forward)

      # Run warmup.
      warmup_latencies = []
      for i in range(warmup_iterations):
        start = time.perf_counter()
        jax.block_until_ready(jit_function(*jit_inputs))
        end = time.perf_counter()
        latency = 1000 * (end - start)
        if i == 0:
          compile_time_s = latency / 1000
        warmup_latencies.append(latency)

      # Run benchmark.
      latencies = []
      for i in range(benchmark_iterations):
        start = time.perf_counter()
        outputs = jit_function(*jit_inputs)
        outputs.block_until_ready()
        end = time.perf_counter()
        latencies.append(1000 * (end - start))
        is_equal = np.allclose(outputs, expected_outputs[0], atol=1e-5)
        if not is_equal:
          max_diff = np.max(np.abs(expected_outputs[0] - outputs))
          raise RuntimeError(f"Outputs do not match expected. Max diff: {max_diff}")

      # Save results.
      result_dict = {
          "min_warmup_latency_ms": min(warmup_latencies, default=None),
          "max_warmup_latency_ms": max(warmup_latencies, default=None),
          "mean_warmup_latency_ms": None if not warmup_latencies else statistics.mean(warmup_latencies),
          "median_warmup_latency_ms": None if not warmup_latencies else statistics.median(warmup_latencies),
          "stddev_warmup_latency_ms": None if not warmup_latencies else statistics.stdev(warmup_latencies),
          "warmup_iterations": warmup_iterations,
          "min_latency_ms": min(latencies, default=None),
          "max_latency_ms": max(latencies, default=None),
          "mean_latency_ms": None if not latencies else statistics.mean(latencies),
          "median_latency_ms": None if not latencies else statistics.median(latencies),
          "stddev_latency_ms": None if not latencies else statistics.stdev(latencies),
          "benchmark_iterations": benchmark_iterations,
          "compile_time_s": compile_time_s,
          "input_data_transfer_ms": input_data_transfer_ms,
      }
      shared_dict.update(result_dict)

  except Exception as e:
    print(f"Failed to benchmark model {model_name}. Exception: {e}")
    raise e


if __name__ == "__main__":
  argParser = argparse.ArgumentParser()
  argParser.add_argument(
      "-o",
      "--output_path",
      help=
      "Path to results json file. Expects this file to have been pre-populated."
  )
  argParser.add_argument("-bid",
                         "--benchmark_id",
                         help="The unique id that defines a benchmark.")
  argParser.add_argument("-w",
                         "--warmup_iterations",
                         type=int,
                         default=5,
                         help="The number of warmup steps.")
  argParser.add_argument("-iter",
                         "--iterations",
                         type=int,
                         default=100,
                         help="The number of iterations to benchmark.")
  argParser.add_argument(
      "-d",
      "--device",
      default="gpu",
      help="The device to run on. Currently `cpu` and `gpu` are supported.")
  argParser.add_argument("--hlo_benchmark_path",
                         default=None,
                         help="The path to `run_hlo_module`.")
  argParser.add_argument(
      "--run_in_process",
      action="store_true",
      help=
      "Whether to run the benchmark under the same process. Set this to true when profiling a single workload"
  )
  argParser.add_argument("--cache_dir",
                         required=True,
                         type=pathlib.Path,
                         help="Directory to download artifacts to.")

  args = argParser.parse_args()

  model_name, model_class, model_definition = benchmark_lookup(
      args.benchmark_id)
  print(
      f"\n\n--- {model_name} {args.benchmark_id} -------------------------------------"
  )

  benchmark_definition = {
      "benchmark_id": args.benchmark_id,
      "benchmark_name": model_definition.name,
      "framework": str(model_definition.meta_model.framework_type),
      "data_type": str(model_definition.meta_model.data_type),
      "batch_size": model_definition.input_batch_size,
      "inputs": model_definition.inputs.tensor_dimensions,
      "outputs": model_definition.outputs.tensor_dimensions,
      "compiler": "xla",
      "device": args.device,
      "tags": model_definition.meta_model.tags + model_definition.tags,
  }

  inputs = retrieve_data(model_definition.inputs, args.cache_dir)
  expected_outputs = retrieve_data(model_definition.outputs, args.cache_dir)

  framework_metrics = {}
  # Retrieve framework-level benchmarks.
  with multiprocessing.Manager() as manager:
    shared_dict = manager.dict()

    if args.run_in_process:
      run_framework_benchmark(model_name, model_class, inputs, expected_outputs, args.warmup_iterations,
                              args.iterations, args.device, shared_dict)
    else:
      p = multiprocessing.Process(target=run_framework_benchmark,
                                  args=(model_name, model_class, inputs, expected_outputs,
                                        args.warmup_iterations, args.iterations,
                                        args.device, shared_dict))
      p.start()
      p.join()

    framework_metrics.update(shared_dict)

  result = {
      "definition": benchmark_definition,
      "metrics": {
          "framework_level": framework_metrics,
      }
  }
  print(json.dumps(result, indent=2))
  dump_result(args.output_path, result)
