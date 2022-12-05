# Production-RWKV

RWKV is a Sequence to Sequence Model that takes the best features of GPT and (Recurrent Nueral Network (RNN)) that performs Language Model (LM). That is used to generate tex Auto Regressively (AR).
It has Transformer Level Performance without the quadratic attention mechanism. It borrows ideas from Attention Free Transformers, meaning the attention is a linear in complexity. Allowing for infinite context windows.

More from the Research and Development Repository:
https://github.com/BlinkDL/RWKV-LM

RWKV - Receptance Weighted Key Value 

This project aims to make RWKV Accessible to everyone using a Hugging Face-esque OOP interface. 

Q: So why not port it to Hugging Face? 
A: Well as of right now RWKV goes through many changes and is involved in Research, 
the Hugging Face framework is large and vast and requires a lengthy PR process that maybe ignored for long peroids of time.
This project aims to port the latest developments in the RWKV make them super accessible with 3 lines of code.
While keeping it close to the R and D RWKV branch of code. This is a very thin layer over the core features of RWKV.

# API 

```python
from rwkvtokenizer import RWKVTokenizer
from rwkvrnnmodel import RWKVRNN4NeoForCausalLM

tokenizer = RWKVTokenizer.from_file()
model = RWKVRNN4NeoForCausalLM.from_pretrained()

context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
context_input_ids = tokenizer.encode(context).ids


def callback(ind):
    index = tokenizer.decode([ind],skip_special_tokens=False)
    print(ind)
    # print(repr(index),end="")

ctx = model.generate(inputs=context_input_ids,streaming_callback=callback,max_length=512)

print(repr(tokenizer.decode(ctx,skip_special_tokens=False))) # cpu 3 tokens a second

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

# Road Map
Provide the performance numbers for the models in various deployment strategies focusing on deployment to Edge Devices. 

- Onnx CUDA / CPU
- Torch CUDA / CPU
- Torch.jit CUDA / CPU

Provide Notebook
Provide Pip Package
Remove HF Tokenizers Dependency
