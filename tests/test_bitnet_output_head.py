"""Exercise actual Bitnet tensor writing with tiny NumPy-backed tensors."""
import ast
from enum import IntEnum
import logging
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

source = Path(__file__).resolve().parents[1] / "utils/convert-hf-to-gguf-bitnet.py"
tree = ast.parse(source.read_text())
cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "BitnetModel")
method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "write_tensors")
quantizer = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "quantize_to_i2_s")
module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), method, quantizer], type_ignores=[])
ast.fix_missing_locations(module)
class QType(IntEnum):
    F32 = 0
    F16 = 1
    I2_S = 36
    TL1 = 37
    TL2 = 38
class Tensor:
    def __init__(self, array): self.array = array; self.dtype = array.dtype
    def squeeze(self): return self
    def numpy(self): return self.array
    def to(self, dtype): return Tensor(self.array.astype(dtype))
class Writer:
    def __init__(self): self.tensors = {}
    def add_tensor(self, name, array, **kwargs): self.tensors[name] = (array, kwargs)
ns = {"np": np, "logger": logging.getLogger("test"), "args": SimpleNamespace(quant_embd=False), "torch": SimpleNamespace(float16=np.float16, float32=np.float32), "gguf": SimpleNamespace(GGMLQuantizationType=QType, MODEL_TENSOR=SimpleNamespace(FFN_GATE_INP=1, POS_EMBD=2, TOKEN_TYPES=3, TOKEN_EMBD=4))}
exec(compile(module, str(source), "exec"), ns)

class OutputHeadTests(unittest.TestCase):
    def test_untied_head_is_preserved_while_projection_is_quantized(self):
        head = np.linspace(-0.7, 0.8, 128, dtype=np.float32).reshape(8, 16)
        projection = np.ones((8, 16), dtype=np.float32)
        tensors = [("lm_head.weight", Tensor(head)), ("model.layers.0.self_attn.q_proj.weight", Tensor(projection))]
        writer = Writer()
        names = {"lm_head.weight": "output.weight", "model.layers.0.self_attn.q_proj.weight": "blk.0.attn_q.weight"}
        model = SimpleNamespace(ftype=QType.I2_S, tensor_map=SimpleNamespace(mapping={"output": (None, "output")}), get_tensors=lambda: iter(tensors), modify_tensors=lambda data, name, bid: [(names[name], data)], match_model_tensor_name=lambda *args: False, gguf_writer=writer)
        ns["write_tensors"](model)
        data, metadata = writer.tensors["output.weight"]
        self.assertEqual(metadata["raw_dtype"], QType.F16)
        np.testing.assert_array_equal(data, head.astype(np.float16))
        self.assertEqual(writer.tensors["blk.0.attn_q.weight"][1]["raw_dtype"], QType.I2_S)

if __name__ == "__main__": unittest.main()
