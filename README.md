# Production-RWKV
RWKV - Receptance Weighted Key Value 

* RWKV is a Sequence to Sequence Model that takes the best features of Generative PreTraining (GPT) and Recurrent Nueral Networks (RNN) that performs Language Modelling (LM). 
This is used to generate text Auto Regressively (AR).

* It has Transformer Level Performance without the quadratic attention mechanism. 
It borrows ideas from Attention Free Transformers, meaning the attention is a linear in complexity. 
Allowing for infinite context windows.

More from the Research and Development Repository:
https://github.com/BlinkDL/RWKV-LM

This project aims to make RWKV Accessible to everyone using a familiar interface similar to Hugging Face. 

* Q: So why not port it to Hugging Face?

* A: Well as of right now RWKV goes through many changes and is involved in Research, 
the Hugging Face framework is large and vast and requires a lengthy PR process that maybe ignored for long peroids of time.
This project aims to port the latest developments in the RWKV make them super accessible with few lines of code.
While keeping it close to the R and D RWKV branch of code. 
This is a very thin layer over the core features of RWKV.

# API 

```python
from prwkv.rwkvtokenizer import RWKVTokenizer
from prwkv.rwkvrnnmodel import RWKVRNN4NeoForCausalLM

tokenizer = RWKVTokenizer.default()
model = RWKVRNN4NeoForCausalLM.from_pretrained("/Users/michaelchung/Code/Production-RWKV/RWKV-4-Pile-430M-20220808-8066",n_layer=24,n_embd=1024,ctx_len=1024)

context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
context_input_ids = tokenizer.encode(context).ids

def callback(ind):
    token = tokenizer.decode([ind],skip_special_tokens=False)
    print(token,end="")
ctx = model.generate(inputs_id=context_input_ids,streaming_callback=callback,max_length=128)
result = tokenizer.decode(ctx,skip_special_tokens=False) # cpu 3 tokens a second
print(f"\n---Result---:\n{result}")

```

```python
# using latest pretrained from hugging face
tokenizer = RWKVTokenizer.default()
model = RWKVRNN4NeoForCausalLM.from_pretrained("RWKV-4-430M") # options RWKV-4-1B5  RWKV-4-7B  RWKV-4-14B
```

There are two models for RWKV.

* RWKV_RNN: This model is for running inference quickly. ( Availible for Inference )

* RWKV (RWKV GPT): This model is for training or fine tuning your model quickly. ( Not Availible For Training Yet )

The checkpoints for the models can be used for both models.

* Another special note about RWKV-LM is that you can use RWKV GPT as an context encoder to generate the context for the decoder very simular to cross attention mechanism with Encoder Decoder Architectures. This will be implemented at a future date. As it requires weight sharing.

# How to install

```
pip3 install PRWKV
```

# Road Map
* [ ] Provide Pip Package
* [ ] Zero Shot Bench Mark Test
Provide the performance numbers for the models in various deployment strategies focusing on deployment to Edge Devices iPhone and Android. 

* [ ] Onnx CUDA / CPU Performance Charts and Export Scripts
* [ ] Torch CUDA / CPU and Streaming Large Models | Performance Charts | Export Scripts
* [ ] Torch.jit CUDA / CPU and Streaming Large Models | Performance Charts | Export Scripts

* [ ] Beam Search
* [ ] Typical Sampling
* [ ] Tail Free Sampling
* [ ] Top-p-x sampling method 
* [ ] Provide Better Documentation

Train:
* [ ] Seeker Dialog Model
* [ ] Palm Instruction Tuned Model
* [ ] GPT Instruction Tuned Model
