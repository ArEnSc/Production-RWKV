

from scipy.special import softmax
import numpy as np


def notimplemented(*args):
    raise "not implemented"


class RWKVOPS():
    def __init__(self, layers, embed):
        print("init RWKVOPS, from super")
        self.initTensor: notimplemented
        self.initCpuTensor = lambda x: self.initTensor(x)
        self.sqrt: notimplemented
        self.mean: notimplemented
        self.relu: notimplemented
        self.exp: notimplemented
        self.add = lambda x, y: x+y
        self.divide = lambda x, y: x/y
        self.multiply = lambda x, y: x*y
        self.subtract = lambda x, y: x-y
        self.stack: notimplemented
        self.matvec: notimplemented
        self.layernorm: notimplemented
        self.lerp: notimplemented
       # module def
        self.module: notimplemented
        self.log: notimplemented
        self.minimum: notimplemented
        self.klimit: notimplemented
       # tensorflow function defs
        self.initfunc: notimplemented
        self.layerdef: notimplemented
        self.mainfunc: notimplemented
        self.prefunc: notimplemented
        self.postfunc: notimplemented
        self.emptyState: notimplemented
        self.logistical = lambda x: 1 / (self.exp(x) + 1)
        self.postProcessModule = lambda x: x

        def sample(ozut, temp: float = 1.0, top_p_usual: float = 0.8) -> int:
            try:
                ozut = ozut.numpy()
            except:
                try:
                    ozut = ozut.cpu().numpy()
                except:
                    ozut = np.array(ozut)
            # out[self.UNKNOWN_CHAR] = -float('Inf')
            # out[self.UNKNOWN_CHAR] = -float('Inf')
            # turn to float if is half and cpu
            probs = softmax(ozut, axis=-1)

            sorted_probs = np.sort(probs)[::-1]
            cumulative_probs = np.cumsum(sorted_probs)
            cutoff = float(sorted_probs[np.argmax(
                cumulative_probs > top_p_usual)])
            probs[probs < cutoff] = 0
            if temp != 1.0:
                probs = pow(probs, 1.0 / temp)
            probs = probs / np.sum(probs, axis=0)
            mout = np.random.choice(a=len(probs), p=probs)
            return mout

        self.sample = sample

        # typing, set as any
        self.tensorDef = None
