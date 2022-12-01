
import wget
import os
from urllib.parse import urlparse
from pathlib import Path
import gc
from typing import List,Union,Callable
import types
import torch
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

class RWKV_RNN_Model():
    def __init__(self,context_length:int,
                number_of_layers:int,
                embedding_dim:int,
                file_path:Path,
                eos_token_id:int = 0,
                padd_token_id:int = 1,
                device_type:str="cpu", # cpu or cuda
                ):
        
        self.context_length = context_length
        self.number_of_layers = number_of_layers
        self.embedding_dim  = embedding_dim
        self.eos_token_id = eos_token_id
        self.padd_token_id = padd_token_id

        from modelrun import RWKV_RNN
        
        self.args = types.SimpleNamespace()
        self.args.RUN_DEVICE = device_type
        self.args.MODEL_NAME = file_path

        self.args.FLOAT_MODE = "fp32" # fp32 (good for cpu) // fp16 (might overflow) // bf16 (less accurate)
        self.args.n_layer = self.number_of_layers
        self.args.n_embd = self.embedding_dim
        self.args.vocab_size = 50277
        self.args.head_qk = 0
        self.args.pre_ffn = 0
        self.args.grad_cp = 0
        self.args.my_pos_emb = 0
        os.environ["RWKV_RUN_DEVICE"] = self.args.RUN_DEVICE

        self.model = RWKV_RNN(self.args)

        self.init_state = None
        self.init_logits = None
    def _warmup_with_context(self,context:List[int],save_path:Path):
        init_state = None 
        context_length = len(context)

        for i in range(context_length):
            x = context[: i + 1]
            if i == context_length - 1:
                logits, init_state = self.model.forward(x, init_state)
            else:
                init_state = self.model.forward(x, init_state, preprocess_only=True)

        gc.collect()
        self.init_state = init_state.clone()
        self.init_logits = logits.clone()

    def generate(self,
                 inputs,
                 streaming_callback:Callable[[int], None], # streams single word
                 bad_words_ids=[],
                 force_words_ids=[],
                 min_length=0,
                 max_length=128,
                 early_stopping=False,
                 temperature=1.0,
                 top_p=.9,
                 top_k=5,
                 repetition_penalty=2.5,
                 do_sample=False)->Union[List[int],None]: 
                 
            context = inputs

            state = None
            for i in range(max_length):
                if i == 0:
                    # greedy
                    state = self.init_state
                    token = torch.argmax(self.init_logits)
                else:
                    logits, new_state = self.model.forward(context, state)

                    state = new_state
                    # greedy
                    token = torch.argmax(logits)
                
                context.append(token)
                if streaming_callback != None:
                    streaming_callback(token)
                
            return context

import sys
def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

class RWKVRNN4NeoForCausalLM():

    @staticmethod
    def from_pretrained(file_path_or_url:str="https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-20220929-ctx4096.pth",cache_folder_path:Path=Path(".")):
        # download the model using wget from huggingface read json file lol doesn't exist yet.
        # check for model configuration from json file in the future right now just bake it in until someone writes the json files.
        n_layer = None
        n_embd = None
        ctx_len = None
        # hack for now to keep everything working
        if file_path_or_url == "https://huggingface.co/BlinkDL/rwkv-4-pile-1b5/resolve/main/RWKV-4-Pile-1B5-20220929-ctx4096.pth":
            n_layer = 24
            n_embd = 2048
            ctx_len = 4096
        else:
            assert(False)

        # if it is a url download
        if urlparse(url=file_path_or_url).scheme != "":
            url = file_path_or_url

            files = os.listdir(cache_folder_path)
            # filter by ending in .pth
            files = [f for f in files if f.endswith(".pth")]

            file = url 
            file = file.split("/")[-1]
            
            if not file in os.listdir():
                # download with wget
                file = wget.download(url=url,out=cache_folder_path,bar=bar_progress)
            # model file name
            # use pathlib lol
            file = file.split('.')[0]
            model_file_path = str(Path(cache_folder_path,file))
        else:
            model_file_path = Path(file_path_or_url)
            
        # configure model
        model = RWKV_RNN_Model(context_length=ctx_len,number_of_layers=n_layer,embedding_dim=n_embd,file_path=model_file_path)
        return model
