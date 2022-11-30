# Production-RWKV

RWKV is a Sequence to Sequence Model (Recurrent Nueral Network (RNN)) that performs Language Model (LM). That is used to generate tex Auto Regressively (AR).
It has Transformer Level Performance without the quadratic attention mechanism.

More from the Research and Development Repository:
https://github.com/BlinkDL/RWKV-LM

RWKV - Receptance Weighted Key Value 

This project aims to make RWKV Accessible to everyone using a Hugging Face like interface. 
Provide the performance numbers for the models in various deployment strategies focusing on deployment to Edge Devices. 

- Onnx CUDA / CPU
- Torch CUDA / CPU
- Torch.jit CUDA / CPU

Q: So why not port it to Hugging Face? 
A: Well as of right now RWKV goes through many changes and is involved in Research, 
the Hugging Face framework is large and vast and requires a lengthy PR process that maybe ignored for long peroids of time.
This project aims to port the latest developments in the RWKV make them super accessible with 3 lines of code.
While keeping it close to the R and D RWKV branch of code. This is a very thin layer over the core features of RWKV.

# API 
Looks like this:

```python
tokenizer = PRWKVToeknizer()
model = RWKV4MForCausalLM("CheckpointPathLocalOrRemote")
input_ids = tokenizer.encode("Hello World")["input_ids"]
sequence,scores = model.generate(input_ids...) # more params in the docs
tokenizer.decode(output_ids)
```

There are two models for RWKV
RWKV_RNN: This model is for running inference quickly.
RWKV (RWKV GPT): This model is for training or fine tuning your model quickly.
The checkpoints for the models can be used for both models.

Another special note about RWKV-LM is that you can use RWKV GPT as an context encoder to generate the context for the decoder very simular to cross attention mechanism with Encoder Decoder Architectures.

# How to install
```
pip3 install PRWKV
```
