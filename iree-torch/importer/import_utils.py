import io
from typing import Any, Tuple
import torch
import torch_mlir

from torch._decomp import get_decompositions
from torch.fx.experimental.proxy_tensor import make_fx


def _strip_overloads(gm):
  """Modifies the target of graph nodes in :attr:`gm` to strip overloads.
    Args:
        gm(fx.GraphModule): The input Fx graph module to be modified
    """
  for node in gm.graph.nodes:
    if isinstance(node.target, torch._ops.OpOverload):
      node.target = node.target.overloadpacket
  gm.recompile()


def _transform_fx(fx_g):
  kwargs_dict = {
      "dtype": torch.float16,
      "device": torch.device(type="cpu"),
      "pin_memory": False,
  }
  for node in fx_g.graph.nodes:
    if node.op == "call_function":
      if node.target in [torch.ops.aten.arange, torch.ops.aten.empty, torch.ops.aten.zeros]:
        node.kwargs = kwargs_dict

      # Inputs and outputs of aten.var.mean should be upcasted to fp32.
      if node.target in [torch.ops.aten.var_mean]:
        with fx_g.graph.inserting_before(node):
          new_node = fx_g.graph.call_function(
              torch.ops.prims.convert_element_type,
              args=(node.args[0], torch.float32),
              kwargs={},
          )
          node.args = (new_node, node.args[1])
      
      if node.name.startswith("getitem"):
        with fx_g.graph.inserting_before(node):
          if node.args[0].target in [torch.ops.aten.var_mean]:
            new_node = fx_g.graph.call_function(
                torch.ops.aten._to_copy,
                args=(node,),
                kwargs={"dtype": torch.float16},
            )
            node.append(new_node)
            node.replace_all_uses_with(new_node)
            new_node.args = (node,)
            new_node.kwargs = {"dtype": torch.float16}
      
      # aten.empty should be filled with zeros.
      if node.target in [torch.ops.aten.empty]:
        with fx_g.graph.inserting_after(node):
          new_node = fx_g.graph.call_function(
              torch.ops.aten.zero_,
              args=(node,),
          )
          node.append(new_node)
          node.replace_all_uses_with(new_node)
          new_node.args = (node,)

  fx_g.graph.lint()


def import_torch_module(module: torch.nn.Module, inputs: Tuple[Any, ...],
                        output_dialect: torch_mlir.OutputType):
  mlir_module = torch_mlir.compile(module, inputs, output_type=output_dialect)
  bytecode_stream = io.BytesIO()
  mlir_module.operation.write_bytecode(bytecode_stream)
  return bytecode_stream.getvalue()


def import_torch_module_with_fx(module: torch.nn.Module, inputs: Tuple[Any,
                                                                       ...],
                                output_dialect: torch_mlir.OutputType,
                                use_fp16=False):
  fx_g = make_fx(
      module,
      decomposition_table=get_decompositions([
          torch.ops.aten.embedding_dense_backward,
          torch.ops.aten.native_layer_norm_backward,
          torch.ops.aten.slice_backward,
          torch.ops.aten.select_backward,
          torch.ops.aten.norm.ScalarOpt_dim,
          torch.ops.aten.native_group_norm,
          torch.ops.aten.upsample_bilinear2d.vec,
          torch.ops.aten.split.Tensor,
          torch.ops.aten.split_with_sizes,
          torch.ops.aten.native_layer_norm,
      ]),
  )(*inputs)

  fx_g.graph.set_codegen(torch.fx.graph.CodeGen())
  fx_g.recompile()

  _strip_overloads(fx_g)

  if use_fp16:
    fx_g = fx_g.half()
    _transform_fx(fx_g)
    fx_g.recompile()

  ts_graph = torch.jit.script(fx_g)
  return import_torch_module(ts_graph, inputs, output_dialect)
