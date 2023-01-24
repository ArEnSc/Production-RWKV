# Changes
# 0.1.9
* Fixed issues with floating point half and cuda half modes. 7B runs on 14 GB 
# 0.1.6
* added option to not include original context in generation
# 0.1.5 
* Fixed Examples.
# 0.1.4
* Bug Fixes can now properly resume the state generation, there was an issue with the array not being copied.
# 0.1.2 
* Bug Fixes can now use float 16 on GPU 
* CPU remains at float 32
# 0.1.1
* Reduce the logging noise in the library.
# 0.1.0 
* Provides Stability and State Management Documentation to be provided at a later date.
# Production-RWKV
RWKV - Receptance Weighted Key Value 

* RWKV is a Sequence to Sequence Model that takes the best features of Generative PreTraining (GPT) and Recurrent Neural Networks (RNN) that performs Language Modelling (LM). 
This is used to generate text Auto Regressively (AR).

* It has Transformer Level Performance without the quadratic attention mechanism. 
It borrows ideas from Attention Free Transformers, meaning the attention is a linear in complexity. 
Allowing for infinite context windows.

More from the Research and Development Repository:
https://github.com/BlinkDL/RWKV-LM

This project aims to make RWKV Accessible to everyone using a familiar interface similar to Hugging Face. 

* Q: So why not port it to Hugging Face? 

* A: Well as of right now RWKV goes through many changes and is involved in Research, 
the Hugging Face framework is large and vast and requires a lengthy PR process that maybe ignored for long periods of time.
This project aims to port the latest developments in the RWKV make them super accessible with few lines of code.
While keeping it close to the R and D RWKV branch of code. 
This is a very thin layer over the core features of RWKV.

Update: I am working on the HF version.

# API 

```python
from prwkv.rwkvtokenizer import RWKVTokenizer
from prwkv.rwkvrnnmodel import RWKVRNN4NeoForCausalLM
tokenizer = RWKVTokenizer.default()
model = RWKVRNN4NeoForCausalLM.from_pretrained("/Users/michaelchung/Code/Production-RWKV/RWKV-4-Pile-430M-20220808-8066",number_of_layers=24,embedding_dimension=1024,context_length=1024)

context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
context_input_ids = tokenizer.encode(context).ids
model.warmup_with_context(context=context_input_ids)

import time

def callback(ind):
    token = tokenizer.decode([ind],skip_special_tokens=False)
    print(token,end="")

# repetition penalty and temperature are senstive and can cause generations to offshoot.
start = time.time()
ctx = model.generate(input_ids=[],streaming_callback=callback,max_length=32,repetition_penalty=0.0,temperature=0.8,stop_on_eos=True)
end = time.time()

print(f"{end-start} seconds \n")

result = tokenizer.decode(ctx,skip_special_tokens=False) # cpu 3 tokens a second
print(f"\n---Result---:\n{result}")

```

```python
# using latest pretrained from hugging face
tokenizer = RWKVTokenizer.default()
model = RWKVRNN4NeoForCausalLM.from_pretrained("RWKV-4-430M") # options RWKV-4-1B5 RWKV-4-3B RWKV-4-7B  RWKV-4-14B
```

# How to install
```
pip3 install prwkv
```

# Quickly Grokking the RWKV Model
So when I first read about this model on less wrong[https://www.lesswrong.com/posts/K4urTDkBbtNuLivJx/why-i-think-strong-general-ai-is-coming-soon] Then spent quite a bit of time digging. The important information seemed to be scattered around the discord and locked up behind the mind of a genius, so this is an attempt to simplify and clarify and surface the ideas and model and how it works.

There are two models for RWKV, they are refered to as modes, specifically in RWKVv4[https://github.com/BlinkDL/RWKV-LM/tree/main/RWKV-v4neo] Folder

This was due discovery in an algebraic formulation for the RWKV_RNN model that allows it to be reformulated as a GPT model (RWKV GPT) with a Self Attention. 

* What makes this `very special` is that weights can be shared and loaded between the two models, allowing an interop between both GPT mode and RNN Mode. This implies you can use both models at the same time because you can share the weights in memory. More on this idea later, as we need to get into the specifics properties of each model mode.

* RWKV_RNN: 
  * This model is designed for running inference quickly. ( Available for Inference in this Package )
  * It has a hidden state that stays a constant size. This hidden state encodes and compresses `the prompt context` and subsequent additions to the prompt context. This means that we really don't need to keep the `prompt context and the history of that in memory like that of a vanilla transformer` because it is encoded in the hidden state. This feature has limitations however... and entirely depends on the context length of the training samples when training using RWKV GPT. Also depends on the floating point accuracy.
  * Blink DL mentioned that when training with GPT Mode with a context length of 1024, he noticed that RWKV_RNN deteriorated around a context length of 2000 so it can extrapolate and compress the `the prompt context` a bit further. This is due to the fact that the model likely doesn't know how to handle samples beyond that size. This implies that the hidden state allows for the `the prompt context` to be infinite, if we can fine tune it properly. 
( Unclear right how ) 

BlinkDL Mentioned 
```
If you train RWKV using the correct method (GPT mode with ctxlen 1024 but apply smart "continuation" of hidden state) to let it learn longer ctxlen, the RNN mode can easily support ctxlen of at least 100k. 
```

* RWKV (RWKV GPT): This mode is for training or fine tuning your model quickly. ( Not Available For Training Yet In this Repo )
  * This mode is designed for training and generating the initial hidden state quickly when in memory weight sharing.
  * The limitation of this mode is that it doesn't contain a hidden state that can't hold an infinite context length.
  * The pros of this mode is that it can utilize parallelism to quickly train because it is in it's GPT configuration.
  * Another pro of this mode is that it contains a linear self attention mechanism allowing for large context lengths.
  * Wierdly RWKV can be trained as an RNN as well ( mentioned in a discord discussion but not implemented )

The checkpoints for the models can be used for both models.
This allows you to transition between both a GPT like model and a RNN like model. Almost like a shape shifting model.

* Another special note about RWKV-LM is that you can use RWKV GPT as an context encoder to generate the context for the decoder very similar to the cross attention mechanism with Encoder Decoder Architectures. This will be implemented at a future date. As it requires in memory weight sharing.

Performance:
| CPU M1 Pro | RWKV-430m fp32 | RWKV-1B5 fp32 | RWKV-3B | RWKV-7B | RWKV-14B |
|--           |--             |--              |--       |--       |--       |
|Tokens/Second| 17-18 Tokens | 4-5 Tokens | NA | NA     | NA      | NA        |
|Memory RAM   |    ~1.3-2GB | ~5.6-5.8 GB | NA | NA     | NA      | NA        |

Performance 3090 (Non Cuda Might need to revisit these metrics):
| GPU 3090 24GB  | RWKV-170m (RWKV-4a-Pile-170M-20221209-7955) fp16 | RWKV-430m (RWKV-4-Pile-430M-20220808-8066) fp16 | RWKV-1B5 (RWKV-4-Pile-1B5-20220929-ctx4096) fp16 | RWKV-3B (RWKV-4-Pile-3B-20221110-ctx4096) fp16 | RWKV-7B (RWKV-4-Pile-7B-20221123-ctx2048) fp16 | RWKV-14B fp16 |
|--              |--                                                |--              |--              |--           |--            |--             |
| 25 Tokens | 0.6221s |  0.9178s | 0.8562s |  1.0058s | 1.0309s | x |
| Memory VRAM  | 900MB |  1.5GB | 3.5GB | 6GB | 14GB | x |
| Warm Up Time | 0.7686s | 0.9178s |  0.8562s |  1.0058s | 1.0309s | x |
| Load Time |  1.9397s | 3.0567s |  6.3156s |  14.0923s | 26.1861s | x | 

| Model | Threads | Load Time | Initialize Time | Run Time (25 tokens) | VRAM/RAM |
|-------|---------|----------|----------------|----------------------|----------|
| RWKV-4-Pile-14B-20221128-3061 | 3090 & 3080Ti | 232.76s | 3.66s | 3.66s | 29199MB |
| RWKV-4-Pile-14B-20221128-3061 | 38 | 218.46s | 40.33s | 60.73s | 54341MB |
| RWKV-4-Pile-1B5-20220929-ctx4096 | 3090 & 3080Ti | 32.42s | 1.36s | 2.08s | 2936MB |
| RWKV-4-Pile-1B5-20220929-ctx4096 | 38 | 3.8s | 3.55s | 5.2s | 5302MB |
| RWKV-4a-Pile-170M-20221209-7955 | 3090 & 3080Ti | 3.92s | 0.58s | 0.92s | 286MB |
| RWKV-4a-Pile-170M-20221209-7955 | 38 | 0.81s | 0.59s | 1.12s | 604MB |
| RWKV-4-Pile-430M-20220808-8066 | 3090 & 3080Ti | 9.02s | 1.22s | 1.86s | 748MB |
| RWKV-4-Pile-430M-20220808-8066 | 38 | 1.42s | 1.53s | 2.56s | 1507MB |
| RWKV-4-Pile-7B-20221123-ctx2048 | 3090 & 3080Ti | 154.28s | 1.79s | 2.97s | 13620MB |
| RWKV-4-Pile-7B-20221123-ctx2048 | 38 | 37.79s | 30.65s | 37.27s | 29667MB |
| RWKV-4-Pile-3B-20221110-ctx4096 | 3090 & 3080Ti | 61.79s | 1.83s | 2.85s | 5700MB |
| RWKV-4-Pile-3B-20221110-ctx4096 | 38 | 7.39s | 11.55s | 15.1s | 11797MB |

# Road Map
* [x] Provide Pip Package
* [ ] Zero Shot Benchmark Test
Provide the performance numbers for the models in various deployment strategies focusing on deployment to Edge Devices iPhone and Android. 

* [ ] Onnx CUDA / CPU Performance Charts and Export Scripts
* [ ] Torch CUDA / CPU and Streaming Large Models | Performance Charts | Export Scripts
* [ ] Torch.jit CUDA / CPU and Streaming Large Models | Performance Charts | Export Scripts

* [ ] Beam Search
* [ ] Typical Sampling
* [ ] Tail Free Sampling
* [ ] Top-p-x sampling method 
* [ ] Provide Better Documentation WIP

Train:
* [ ] Seeker Dialog Model
* [ ] Palm Instruction Tuned Model
* [ ] GPT Instruction Tuned Model

* AdamW takes 8 bytes per parameter ~~
* each model is roughly 
14G VRAM for a 7B model at fp16
28G VRAM for a 7B model at fp32
