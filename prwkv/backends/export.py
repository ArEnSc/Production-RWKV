from prwkv.backends.jax import RWKVJaxOps, RWKVNumpyOps
from prwkv.backends.tensorflow import RWKVTFExport, RWKVTFOps
from prwkv.backends.torch import RWKVCudaDeepspeedOps, RWKVCudaOps, RWKVPTCompatOps, RWKVPTTSExportOps


RwkvOpList = {
    "tensorflow(cpu/gpu)": RWKVTFOps,
    "pytorch(cpu/gpu)": RWKVCudaOps,
    "numpy(cpu)": RWKVNumpyOps,
    "jax(cpu/gpu/tpu)": RWKVJaxOps,
    "pytorch-deepspeed(gpu)": RWKVCudaDeepspeedOps,
    "export-torchscript": RWKVPTTSExportOps,
    "export-tensorflow": RWKVTFExport,
    "pytorch-compatibility(cpu/debug)": RWKVPTCompatOps,
}

RwkvOpListKeys = list(RwkvOpList.keys())
