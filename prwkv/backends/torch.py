
import inquirer
import torch

from prwkv.backends.rwkvop import RWKVOPS


class RWKVPTOps(RWKVOPS):

    def __init__(self, layers, embed, dtype=None):
        RWKVOPS.__init__(self, layers, embed)
        q = [inquirer.List(
            'type',
            message="Load model with which dtype?",
            choices=[torch.bfloat16, torch.float16, torch.float32, torch.float64])]

        if dtype is None:
            a = inquirer.prompt(q)
            dtype = a['type']
        self.dtype = dtype
        # self.sample = torchsample

        def initTensor(x):
            result = x.to(dtype=self.dtype)

            return result

        self.initTensor = initTensor
        self.initCpuTensor = lambda x: self.initTensor(x).cpu()
        self.klimit = torch.tensor(
            [32] * embed).to(dtype=self.dtype)
        self.minimum = torch.minimum
        self.sqrt = torch.sqrt
        self.mean = torch.mean
        self.relu = torch.relu
        self.stack = lambda x: x
        self.matvec = torch.mv
        # safe log
        self.log = lambda x: torch.complex(x, torch.zeros_like(x)).log()

        self.exp = lambda x: torch.exp(x).to(dtype=self.dtype)
        self.lerp = torch.lerp

        # module def
        self.module = torch.nn.Module

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        self.postfunc = lambda x: lambda *args: x(*args).float()
        self.prefunc = lambda x: x

        # self.postProcessModule = ppm

        def layernorm(x, w, b) -> torch.Tensor:

            return torch.layer_norm(x, w.shape, w, b)
        self.layernorm = layernorm
        self.emptyState = torch.zeros(
            4*layers, embed, dtype=self.dtype)+0.0


# Use when doing weird stuff with onnx
class RWKVPTCompatOps(RWKVPTOps):
    def __init__(self, layers, embed, *args):
        RWKVPTOps.__init__(self, layers, embed, *args)
        self.relu = lambda x: torch.max(x, torch.zeros_like(x))
        self.matvec = lambda x, y: torch.sum(x*y, dim=1)

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln


# Use with cuda
class RWKVCudaOps(RWKVPTOps):
    def __init__(self, layers, embed, *args, useGPU=None, runtimedtype=None, **kwargs):
        super().__init__(layers, embed, *args, **kwargs)

        useGPU = inquirer.confirm(
            "Use GPU?", default=True) if useGPU is None else useGPU

        self.useGPU = useGPU

        if not useGPU:
            return

        runtimedtype = inquirer.prompt([inquirer.List(
            'type',
            message="Dtype for non-matrix ops:",
            choices=[torch.bfloat16, torch.float32, torch.float64])])['type'] if runtimedtype is None else runtimedtype

        self.exp = lambda x: torch.exp(x).to(dtype=runtimedtype)

        self.initTensor = lambda x: x.to(dtype=self.dtype if len(
            x.shape) == 2 else runtimedtype, device='cuda')
        self.initCpuTensor = self.initTensor  # could be used for offload

        self.klimit = self.klimit.to(dtype=runtimedtype, device='cuda')

        self.matvec = lambda x, y: x.mv(
            y.to(self.dtype)).to(runtimedtype)

        self.postfunc = lambda x: lambda *args: x(*args).float()

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln

        self.emptyState = torch.zeros(
            4*layers, embed, dtype=runtimedtype, device="cuda")+0.01


class RWKVCudaDeepspeedOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args):
        super().__init__(layers, embed, *args)

        try:
            import deepspeed
        except:
            raise ImportError("deepspeed not installed")

        self.postProcessModule = lambda x: deepspeed.init_inference(
            x, replace_method='auto', replace_with_kernel_inject=True).module


# Not as good as numpy but can be packaged with torchscript
def torchsample(ozut: torch.LongTensor, temp=1.0, top_p_usual=0.8) -> int:
    # do it in pytorch

    probs = torch.softmax(ozut, dim=-1)
    sorted_probs, indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    cutoff = sorted_probs[torch.argmax(
        cumulative_probs[cumulative_probs > top_p_usual])]
    probs[probs < cutoff] = 0
    if temp != 1.0:
        probs = torch.pow(probs, 1.0 / temp)
    probs = probs / torch.sum(probs, dim=-1)
    mout = torch.multinomial(probs, 1)
    return mout.cpu()


# export a jit model
class RWKVPTTSExportOps(RWKVCudaOps):
    def __init__(self, layers, embed, *args, includeSampler=None):
        super().__init__(layers, embed, *args)
        self.stack = lambda x: torch.stack(x)

        includeSampler = inquirer.confirm(
            "Include sampler?", default=True) if includeSampler is None else includeSampler

        if includeSampler:
            self.sample = torchsample
            self.postfunc = lambda x: lambda *args: self.sample(
                x(*args).float().cpu(), torch.tensor(1), torch.tensor(0.9))

        def exportTorchScript(x):
            torch.jit.save(torch.jit.trace(
                x, (torch.LongTensor([0]), self.emptyState), check_trace=False, strict=False), f"model-{layers}-{embed}-{'sampler' if includeSampler else 'logits'}-{'gpu' if self.useGPU else 'cpu'}-{self.dtype}.pt")
            exit()
        self.postProcessModule = exportTorchScript
