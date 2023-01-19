
import torch
from .backends.interopFileLoader import initTFLiteFile, initTorchScriptFile
from .backends.export import RwkvOpList as Backends
from .rwkvmaster import RWKVMaster
import torch
import gc
from typing import Dict
from tqdm import tqdm
import inquirer
import os

# set torch threads to 8
torch.set_num_threads(8)


def RWKV(Path=None, mode=None, *args, **kwargs) -> RWKVMaster:

    if (Path == None):
        files = os.listdir()
        # filter by ending in .pth
        files = [f for f in files if f.endswith(
            ".pth") or f.endswith(".pt") or f.endswith(".tflite")]

        questions = [
            inquirer.List('file',
                          message="What model do you want to use?",
                          choices=files,
                          )]
        Path = inquirer.prompt(questions)["file"]

    if Path.endswith(".pt"):
        return initTorchScriptFile(Path)
    elif Path.endswith(".tflite"):
        return initTFLiteFile(Path)

    if mode is None:
        mode = inquirer.prompt([inquirer.List('mode',
                                              message="What inference backend do you want to use?",
                                              choices=Backends.keys(),
                                              )])["mode"]

    n_layer = 0

    with torch.no_grad():
        w: Dict[str, torch.Tensor] = torch.load(
            Path, map_location="cpu")
        # refine weights and send to correct device
        keys = list(w.keys())
        for x in keys:
            if '.time_' in x:
                w[x] = w[x].squeeze()

            if '.time_decay' in x:
                w[x] = torch.exp(-torch.exp(w[x].double())
                                 )

            if 'receptance.weight' in x:
                w[x] = -w[x]

            w[x].requires_grad = False

            try:
                if (int(x.split('.')[1])+1 > n_layer):
                    n_layer = int(x.split('.')[1])+1
            except:
                pass

    # store weights in self.w

        keys = list(w.keys())

        preprocess = []

        ops = Backends[mode](
            n_layer, len(w[f"blocks.0.ffn.time_mix_k"]), *args, **kwargs)

        for x in tqdm(list(w.keys())):
            w[x] = ops.initTensor(w[x])

        gc.collect()
        torch.cuda.empty_cache()

        class RWKVTFLayer(ops.module):
            def __init__(self, x):
                super(RWKVTFLayer, self).__init__()

                self.i = x

                self.ln1w = (w[f"blocks.{x}.ln1.weight"])
                self.ln1b = (w[f"blocks.{x}.ln1.bias"])
                self.ln2w = (w[f"blocks.{x}.ln2.weight"])
                self.ln2b = (w[f"blocks.{x}.ln2.bias"])
                self.time_decay = (
                    w[f"blocks.{x}.att.time_decay"])
                self.time_first = (
                    w[f"blocks.{x}.att.time_first"])
                self.kktk = (w[f"blocks.{x}.att.time_mix_k"])
                self.vvtv = (w[f"blocks.{x}.att.time_mix_v"])
                self.rrtr = (w[f"blocks.{x}.att.time_mix_r"])
                self.key = (w[f"blocks.{x}.att.key.weight"])
                self.value = (w[f"blocks.{x}.att.value.weight"])
                self.receptance = (
                    w[f"blocks.{x}.att.receptance.weight"])
                self.outputvv = (
                    w[f"blocks.{x}.att.output.weight"])
                self.time_mix_k_ffn = (
                    w[f"blocks.{x}.ffn.time_mix_k"])
                self.time_mix_r_ffn = (
                    w[f"blocks.{x}.ffn.time_mix_r"])
                self.key_ffn = (w[f"blocks.{x}.ffn.key.weight"])
                self.receptance_ffn = (
                    w[f"blocks.{x}.ffn.receptance.weight"])
                self.value_ffn = (
                    w[f"blocks.{x}.ffn.value.weight"])

            @ ops.layerdef
            def forward(self, x, statea, stateb, statec, stated):
                xy = ops.layernorm(x, self.ln1w, self.ln1b)

                kk = ops.matvec(
                    self.key, ops.lerp(statea, xy, self.kktk))

                v = ops.matvec(self.value, ops.lerp(statea, xy, self.vvtv))

                r = ops.logistical(ops.matvec(
                    self.receptance, ops.lerp(statea, xy, self.rrtr)))

                kt = ops.exp(ops.minimum(
                    ops.add(kk, self.time_first), ops.klimit))
                k = ops.exp(ops.minimum(kk, ops.klimit))

                wrd = ops.divide(
                    ops.add(stateb, ops.multiply(kt, v)), ops.add(statec, kt))
                outb = ops.add(ops.multiply(
                    stateb, self.time_decay), ops.multiply(k*v))
                outc = ops.add(ops.multiply(statec, self.time_decay), k)

                mvv = ops.add(x, ops.matvec(
                    self.outputvv, ops.multiply(r, wrd)))

                ddd = ops.layernorm(mvv, self.ln2w, self.ln2b)

                km = ops.relu(ops.matvec(self.key_ffn, ops.lerp(
                    stated, ddd, self.time_mix_k_ffn)))

                rt = ops.logistical(ops.matvec(self.receptance_ffn, ops.lerp(
                    stated, ddd, self.time_mix_r_ffn)))

                x = ops.add(mvv, ops.multiply(
                    ops.matvec(self.value_ffn, km*km), rt))

                return x, xy, outb, outc, ddd

        class RWKVTFPre(ops.module):
            def __init__(self):
                super(RWKVTFPre, self).__init__()
                self.preprocess = ops.stack(preprocess)

            @ ops.prefunc
            def forward(self, x):
                # invert x to be reversed,
                return ops.layernorm(
                    w["emb.weight"][x[-1]], w["blocks.0.ln0.weight"], w["blocks.0.ln0.bias"])

        matvec = ops.matvec
        layernorm = ops.layernorm

        class RWKVTFPost(ops.module):
            def __init__(self):
                super(RWKVTFPost, self).__init__()

                self.postprocess0 = (w["ln_out.weight"])
                self.postprocess1 = (w["ln_out.bias"])
                self.postprocess2 = (w["head.weight"])

            @ ops.postfunc
            def forward(self, x):

                return matvec(self.postprocess2, layernorm(x, self.postprocess0,
                                                           self.postprocess1))

        class myRWKV(ops.module):
            @ ops.initfunc
            def __init__(self):
                super(myRWKV, self).__init__()
                self.preprocess = RWKVTFPre()
                self.ops = ops

                for i in range(n_layer):
                    self.__dict__[f"layer{i}"] = RWKVTFLayer(i)

                self.postprocess = RWKVTFPost()

            @ ops.mainfunc
            def forward(self, x, state=None):

                # profile usage
                # print("start", len(self.mylayers))

                # with torch.profiler.profile(record_shapes=True, use_cuda=True) as prof:

                if (state is None):
                    state = ops.emptyState

                x = self.preprocess.forward(x)

                statea = state[0::4]
                stateb = state[1::4]
                statec = state[2::4]
                stated = state[3::4]

                ot = []

                # print("start", len(self.mylayers))

                for i in range(n_layer):
                    x, aaa, bbb, ccc, ddd = self.__dict__[f"layer{i}"].forward(
                        x, statea[i], stateb[i], statec[i], stated[i])
                    ot = ot + [aaa, bbb, ccc, ddd]

                x = self.postprocess.forward(x)
                # print(len(ot))

                # display usage

                # print(prof.key_averages().table(
                #     sort_by="cuda_time_total", row_limit=10, top_level_events_only=True))
                # exit()
                return x, ops.stack(ot)

        model = ops.postProcessModule(myRWKV())
        emptyState = ops.emptyState
        initTensor = ops.initTensor

        ret = RWKVMaster(model, emptyState, initTensor, ops.sample)

        return ret


class RWKV_RNN():
    def __init__(self, args):
        model = RWKV(Path=args.MODEL_NAME, mode="pytorch(cpu/gpu)",
                     useGPU="cuda" in self.args.RUN_DEVICE, dtype=torch.bfloat16 if "bf" in self.args.FLOAT_MODE else torch.float32 if "32" in self.args.FLOAT_MODE else torch.float16 if "16" in self.args.FLOAT_MODE else torch.float64)
        self.model = model

    def forward(self, x, state=None):
        self.model.lastToken = x[-1]
        self.model.setState(state)
        output = self.model.forward()
        return output["logits"], output["state"]
