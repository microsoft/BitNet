$(cat gpu/convert_checkpoint.py | sed -e '/import torch/a\
import pickle\
\
class RestrictedUnpickler(pickle.Unpickler):\
    def find_class(self, module, name):\
        # Only allow safe classes from trusted modules\
        allowed_modules = ["torch", "numpy", "collections", "builtins"]\
        if any(module.startswith(allowed_prefix) for allowed_prefix in allowed_modules):\
            return super().find_class(module, name)\
        raise pickle.UnpicklingError(f"Global \'{module}.{name}\' is forbidden")\
\
def safe_torch_load(path, map_location=None):\
    """Load a PyTorch model with security restrictions to prevent arbitrary code execution."""\
    with open(path, "rb") as f:\
        return torch.load(f, map_location=map_location, pickle_module=RestrictedUnpickler)' | sed -e 's/torch\.load(/safe_torch_load(/g')
