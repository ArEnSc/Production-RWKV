
from prwkv.backends.rwkvop import RWKVOPS

import inquirer
import os


class RWKVTFOps(RWKVOPS):
    def __init__(self, layers, embed):
        try:
            import tensorflow as tf
        except:
            inst = inquirer.confirm(
                "Tensorflow not installed, do you want to install it?")
            if inst:
                os.system("pip3 install tensorflow")
                import tensorflow as tf
        if (not inquirer.confirm("Do you want to use GPU?")):
            tf.config.experimental.set_visible_devices([], "GPU")
        tf.config.optimizer.set_jit(True)
        tf.config.optimizer.set_experimental_options(
            {"auto_mixed_precision": True})

        super(RWKVTFOps, self).__init__(layers, embed)
        self.initTensor = lambda x: tf.convert_to_tensor(
            x.float().cpu().numpy())
        self.sqrt = tf.sqrt
        self.mean = tf.reduce_mean
        self.relu = lambda x: tf.maximum(x, tf.zeros_like(x))
        self.minimum = tf.minimum
        self.exp = tf.exp
        self.stack = tf.stack
        self.matvec = tf.linalg.matvec
        self.klimit = tf.convert_to_tensor(
            [30]*embed, dtype=tf.float32
        )
        self.log = tf.math.log
        self.lerp = lambda x, y, z: x*(1-z)+y*z
       # module def
        self.module = tf.Module

       # tensorflow function defs
        self.initfunc = lambda x: x
        self.layerdef = tf.function(
            input_signature=5*[tf.TensorSpec(shape=[None], dtype=tf.float32)], jit_compile=True)
        self.mainfunc = tf.function(input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32), tf.TensorSpec(
            shape=[4*layers, embed], dtype=tf.float32)])
        self.prefunc = tf.function(
            input_signature=[tf.TensorSpec(shape=[1], dtype=tf.int32)], jit_compile=True)
        self.postfunc = tf.function(
            input_signature=[tf.TensorSpec(shape=[embed], dtype=tf.float32)], jit_compile=True)
        self.emptyState = tf.zeros([4*layers, embed], dtype=tf.float32)+0.01

        def ln(x, w, b):
            xee2 = x - self.mean(x)

            x2 = self.sqrt(self.mean(xee2*xee2) + 0.000009999999747378752)

            return w*(xee2/x2) + b

        self.layernorm = ln


class RWKVTFExport(RWKVTFOps):
    def __init__(self, layers, embed, splitmodel=None, exports=None):
        super(RWKVTFExport, self).__init__(layers, embed)
        import tensorflow as tf
        path = f"tfdist/rwkv-{layers}-{embed}/"

        def save(x):
            try:
                try:
                    os.mkdir("tfdist")
                except:
                    pass
                os.mkdir(path)
            except:
                pass
            split = splitmodel if splitmodel is not None else inquirer.prompt([inquirer.Confirm(
                'splitmodel', message="Split model?", default=False)])["splitmodel"]

            q = exports if exports is not None else inquirer.checkbox(message="What to export?", choices=[
                "savedmodel32", "tflite32", "tflite16"])

            if "savedmodel32" in q:
                try:
                    os.mkdir(path+"sm")
                except:
                    pass
                if split:
                    tf.saved_model.save(x.preprocess, path+"sm/pre")
                    tf.saved_model.save(x.postprocess, path+"sm/post")
                    for i, l in enumerate(x.mylayers):
                        tf.saved_model.save(l, path+f"sm/layer{i}")
                else:
                    tf.saved_model.save(x, path+"sm/whole")

            if "tflite32" in q:
                try:
                    os.mkdir(path+"tflite32")
                except:
                    pass
                if split:
                    for i, l in enumerate(x.mylayers):
                        converter = tf.lite.TFLiteConverter.from_concrete_functions(
                            [l.forward.get_concrete_function()])
                        tflite_model = converter.convert()
                        open(path+f"tflite32/layer{i}.tflite",
                             "wb").write(tflite_model)
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [x.preprocess.forward.get_concrete_function()])
                    tflite_model = converter.convert()
                    open(path+f"tflite32/pre.tflite", "wb").write(tflite_model)
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [x.postprocess.forward.get_concrete_function()])
                    tflite_model = converter.convert()
                    open(path+f"tflite32/post.tflite", "wb").write(tflite_model)
                else:
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [x.forward.get_concrete_function()])
                    tflite_model = converter.convert()
                    open(f"model-{layers}-{embed}-32.tflite",
                         "wb").write(tflite_model)

            if "tflite16" in q:
                try:
                    os.mkdir(path+"tflite16")
                except:
                    pass
                if split:
                    for i, l in enumerate(x.mylayers):
                        converter = tf.lite.TFLiteConverter.from_concrete_functions(
                            [l.forward.get_concrete_function()])
                        converter.optimizations = [tf.lite.Optimize.DEFAULT]
                        converter.target_spec.supported_types = [tf.float16]
                        tflite_model = converter.convert()
                        open(path+f"tflite16/layer{i}.tflite",
                             "wb").write(tflite_model)
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [x.preprocess.forward.get_concrete_function()])
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                    tflite_model = converter.convert()
                    open(path+f"tflite16/pre.tflite", "wb").write(tflite_model)
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [x.postprocess.forward.get_concrete_function()])
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                    tflite_model = converter.convert()
                    open(path+f"tflite16/post.tflite", "wb").write(tflite_model)
                else:
                    converter = tf.lite.TFLiteConverter.from_concrete_functions(
                        [x.forward.get_concrete_function()])
                    converter.optimizations = [tf.lite.Optimize.DEFAULT]
                    converter.target_spec.supported_types = [tf.float16]
                    tflite_model = converter.convert()
                    open(f"model-{layers}-{embed}-16.tflite",
                         "wb").write(tflite_model)
            exit()
        self.postProcessModule = save
