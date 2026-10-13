import ast
import ctypes
from contextlib import contextmanager
from pathlib import Path
import types
import unittest
from unittest.mock import patch
import torch
from torch._subclasses.fake_tensor import FakeTensorMode


def native_function():
    tree=ast.parse(Path(__file__).parents[1].joinpath("model.py").read_text())
    fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="bitnet_int8xint2_linear")
    calls=[]
    ns=dict(torch=torch,ctypes=ctypes,bitnet_lib=types.SimpleNamespace(bitlinear_int8xint2=lambda *args:calls.append(args)))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fn],type_ignores=[])),"model.py","exec"),ns)
    return ns[fn.name],calls


class NativeContractTests(unittest.TestCase):
    def tensors(self):
        return [torch.zeros((1,2560),dtype=torch.int8,device="cuda:0"),torch.zeros((2560,640),dtype=torch.int8,device="cuda:0"),torch.ones(1,dtype=torch.bfloat16,device="cuda:0"),torch.ones(4,dtype=torch.bfloat16,device="cuda:0")]

    def invoke(self,fn,tensors):
        with patch.object(torch.cuda,"device",return_value=unittest.mock.MagicMock()),patch.object(torch.cuda,"current_stream",return_value=types.SimpleNamespace(cuda_stream=0)),patch.object(torch.Tensor,"data_ptr",return_value=0):
            return fn(*tensors)

    def test_supported_metadata_reaches_native_call(self):
        fn,calls=native_function()
        with FakeTensorMode():
            result=self.invoke(fn,self.tensors())
            self.assertEqual(result.shape,(1,2560))
            self.assertEqual(result.dtype,torch.bfloat16)
        self.assertEqual(len(calls),1)

    def test_selects_stream_and_launch_context_on_input_device(self):
        fn,calls=native_function()
        active = []
        @contextmanager
        def device_context(device):
            active.append(device)
            try: yield
            finally: active.pop()
        def launch(*args):
            self.assertEqual(active,[torch.device("cuda:1")])
            calls.append(args)
        fn.__globals__["bitnet_lib"].bitlinear_int8xint2=launch
        with FakeTensorMode():
            tensors=[tensor.to("cuda:1") for tensor in self.tensors()]
            with patch.object(torch.cuda,"device",side_effect=device_context) as context,patch.object(torch.cuda,"current_stream",return_value=types.SimpleNamespace(cuda_stream=0)) as stream,patch.object(torch.Tensor,"data_ptr",return_value=0):
                fn(*tensors)
            stream.assert_called_once_with(torch.device("cuda:1"))
            context.assert_called_once_with(torch.device("cuda:1"))
        self.assertEqual(active,[])
        self.assertEqual(len(calls),1)

    def test_rejects_wrong_native_element_types(self):
        fn,calls=native_function()
        with FakeTensorMode():
            for index in range(4):
                tensors=self.tensors();tensors[index]=tensors[index].to(torch.float32)
                with self.assertRaises(TypeError):self.invoke(fn,tensors)
        self.assertEqual(calls,[])

    def test_rejects_strides_ignored_by_native_kernel(self):
        fn,calls=native_function()
        with FakeTensorMode():
            tensors=self.tensors();tensors[1]=torch.zeros((640,2560),dtype=torch.int8,device="cuda:0").t()
            self.assertEqual(tensors[1].shape,(2560,640))
            with self.assertRaisesRegex(ValueError,"contiguous"):self.invoke(fn,tensors)
        self.assertEqual(calls,[])

    def test_rejects_other_device_pointers_before_launch(self):
        fn,calls=native_function()
        with FakeTensorMode():
            for device in ["cpu","cuda:1"]:
                tensors=self.tensors();tensors[2]=torch.ones(1,dtype=torch.bfloat16,device=device)
                with self.assertRaisesRegex(ValueError,"same CUDA"):self.invoke(fn,tensors)
        self.assertEqual(calls,[])

if __name__=="__main__":unittest.main()
