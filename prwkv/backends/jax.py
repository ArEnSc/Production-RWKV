
from prwkv.backends.rwkvop import RWKVOPS
import numpy as np


class RWKVNumpyOps(RWKVOPS):
    def __init__(self, layers, embed):
        super().__init__(layers, embed)
        self.initTensor = lambda x: x.float().cpu().numpy()
        self.sqrt = lambda x: np.sqrt(x)
        self.mean = lambda x: np.mean(x)
        self.relu = lambda x: np.maximum(x, 0)
        self.exp = lambda x: np.exp(x)
        self.stack = lambda x: x
        self.matvec = np.matmul
        self.lerp = lambda x, y, z: x*(1-z) + y*(z)
        self.minimum = lambda x, y: np.minimum(x, y)
        self.klimit = [32] * embed
        # module def
        self.module = object
        self.log = np.log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        self.postfunc = lambda x: x
        self.prefunc = lambda x: x

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b
        self.layernorm = ln
        self.emptyState = [[0.01]*embed]*4*layers


class RWKVJaxOps(RWKVOPS):
    def __init__(self, layers, embed):
        from jax import numpy as npjax
        super().__init__(layers, embed)
        self.initTensor = lambda x: npjax.array(x.float().cpu().numpy())
        self.sqrt = lambda x: npjax.sqrt(x)
        self.mean = lambda x: npjax.mean(x)
        self.relu = lambda x: npjax.maximum(x, 0)
        self.exp = lambda x: npjax.exp(x)
        self.stack = lambda x: x
        self.matvec = npjax.matmul
        self.lerp = lambda x, y, z: x*(1-z) + y*(z)
        self.minimum = lambda x, y: npjax.minimum(x, y)
        self.klimit = npjax.array([32] * embed)
        # module def
        self.module = object
        self.log = npjax.log

        # pytorch function defs
        self.initfunc = lambda x: x
        self.layerdef = lambda x: x
        self.mainfunc = lambda x: x
        # in postfunc, convert to numpy
        self.postfunc = lambda x: lambda self, y: np.array(x(self, y))
        self.prefunc = lambda x: x

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln
        self.emptyState = npjax.array([[0.01]*embed]*4*layers)
