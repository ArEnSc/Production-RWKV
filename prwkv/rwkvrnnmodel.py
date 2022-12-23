
from pathlib import Path
import wget
from urllib.parse import urlparse

import os
import gc
import types
import torch
import json
import sys

from typing import List,Union,Callable
from .modelrun import RWKV_RNN
import torch.nn.functional as F
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True

def bar_progress(current, total, width=80):
    progress_message = "Downloading: %d%% [%d / %d] bytes" % (current / total * 100, current, total)
    # Don't use print() as it will print in new line every time.
    sys.stdout.write("\r" + progress_message)
    sys.stdout.flush()

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
        print(self.args)
        self.model = RWKV_RNN(self.args)

        self.init_state = None
        self.init_logits = None

    def half(self,mode="fp16"):
        import platform
        if platform.system() == "Darwin":
            assert False, "Not Supported"
        elif platform.system() == "Windows":
            self.args.FLOAT_MODE = mode
            self.model = RWKV_RNN(self.args)
        else:
            self.args.FLOAT_MODE = mode
            self.model = RWKV_RNN(self.args)

    def cuda(self):
        self.args.RUN_DEVICE = "cuda"
        self.model = RWKV_RNN(self.args)

    def cpu(self):
        self.args.RUN_DEVICE = "cpu"
        self.model = RWKV_RNN(self.args)

    def clear_memory(self):
        self.init_state = None
        self.init_logits = None 
        
    def warmup_with_context(self,context:List[int],save_path:Path):
        init_state = None 
        context_length = len(context)

        for i in range(context_length):
            x = context[: i + 1]
            if i == context_length - 1:
                logits, init_state = self.model.forward(x, init_state)
            else:
                init_state = self.model.forward(x, init_state, preprocess_only=True)

        gc.collect()
        self.init_state = init_state.detach().clone()
        self.init_logits = logits.detach().clone()

    @staticmethod
    def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):

        assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
        top_k = min(top_k, logits.size(-1))  # Safety check

        if top_k > 0:
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = filter_value

        return logits

    def _warp_logits(self,logits:torch.tensor,
                        temperature:float,
                        top_p:float,
                        top_k:float,
                        repetition_penalty:float,
                        bad_words_ids:List[int],
                        force_words_ids:List[int])->int:
        """
        Args:
            logits (_type_): logits vector
        Returns:
            int: token id
        """
        if temperature == 0:
            # greedy
            token_id = torch.argmax(logits)
            return token_id

        warped_logits = logits / temperature 

        # try different samplers here
        warped_logits = RWKV_RNN_Model.top_k_top_p_filtering(logits=logits,
                                                            top_k=top_k,
                                                            top_p=top_p)

        probabilities = F.softmax(warped_logits, dim=-1)
        # a = probabilities > 0.0
        # indices = a.nonzero()
        # print(indices.shape)

        token_id = torch.multinomial(input=probabilities,num_samples=1)

        return token_id

    def generate(self,
                 inputs_id:List[int],
                 streaming_callback:Callable[[int], None], # streams single word
                 bad_words_ids=[],
                 force_words_ids=[],
                 min_length=0,
                 max_length=128,
                 temperature=.85,
                 top_p=.9,
                 top_k=20,
                 stop_on_eos=False,
                 repetition_penalty=2.5)->Union[List[int],None]:

            assert self.init_state != None,"Use Warm Up function to warm up the context."
            
            context = inputs_id

            # compute the first token using the intial context
            state = self.init_state.detach().clone()# make a copy it's mutable :)
            logits = self.init_logits

            token_id = self._warp_logits(logits=logits,
                                temperature=temperature,
                                top_k=top_k,
                                top_p=top_p,
                                repetition_penalty=repetition_penalty,
                                bad_words_ids=bad_words_ids,
                                force_words_ids=force_words_ids) # 1 by 1 tensor
            
            context.append(token_id[0])

            if streaming_callback != None:
                    streaming_callback(token_id)

            # continue computing the rest of the tokens
            next_token = token_id
            for _ in range(max_length-1): # since we already 
                logits, new_state = self.model.forward(next_token, state)
                state = new_state
                    
                token_id = self._warp_logits(logits=logits,
                                    temperature=temperature,
                                    top_k=top_k,
                                    top_p=top_p,
                                    repetition_penalty=repetition_penalty,
                                    bad_words_ids=bad_words_ids,
                                    force_words_ids=force_words_ids)
                
                context.append(token_id[0])
                
                if streaming_callback != None:
                    streaming_callback(token_id[0])

                if token_id == self.eos_token_id and stop_on_eos:
                    break

                next_token = token_id

            return context


class RWKVRNN4NeoForCausalLM():

    @staticmethod
    def from_pretrained(file_path_or_name:str,
                        number_of_layers:int=None,
                        embedding_dimension:int=None,
                        context_length:int=None,
                        cache_folder_path:Path=Path("./")):
        """
        Loads a RWKVRNN Model
        You can load in two ways
        From file directly, this requires you fill in: 
        number_of_layers:int
        embedding_dimension:int
        context_length:int
        
        ```python
        # example
        model = RWKVRNN4NeoForCausalLM.from_pretrained("path_to_ckpt",number_of_layers=24,embedding_dimension=784,context_length=1024)
        ```

        Or you can load a pretrained model from Hugging Face like so it will use the latest trained model.
        ```python
        
        # where file_path_or_name 

        RWKV-4-169M
        RWKV-4-430M
        RWKV-4-1B5
        RWKV-4-3B
        RWKV-4-7B
        RWKV-4-14B

        model = RWKVRNN4NeoForCausalLM.from_pretrained("RWKV-4-430M")
        ```
        Args:
            file_path_or_name (str): checkpoint path WITHOUT .ckpt extension
            number_of_layers (int, optional): Number of Layers
            embedding_dimension (int, optional): Embedding Dimension
            context_length (int, optional): Context Length
            cache_folder_path (Path, optional): This is the cache path for your pretrained downloaded model. Defaults to Path("").

        Returns:
            _type_: RWKV_RNN_Model a wrapper over RWKV_RNN 
        """

        number_of_layers = number_of_layers
        embedding_dimension = embedding_dimension
        context_length = context_length
        json_dict = None 

        file_path = Path(__file__)
        final_file_path = Path(file_path.parent) / Path("data")
        if file_path_or_name == "RWKV-4-169M":
            with open(file=Path(final_file_path) / Path("RWKV-4-169M.json")) as f:
                json_dict = json.load(f)
        if file_path_or_name == "RWKV-4-430M":
            with open(file=Path(final_file_path) / Path("RWKV-4-430M.json")) as f:
                json_dict = json.load(f)
        elif file_path_or_name == "RWKV-4-1B5":
            with open(file=Path(final_file_path) / Path("RWKV-4-1B5.json")) as f:
                json_dict = json.load(f)
        elif file_path_or_name == "RWKV-4-3B":
            with open(file=Path(final_file_path) / Path("RWKV-4-3B.json")) as f:
                json_dict = json.load(f)
        elif file_path_or_name == "RWKV-4-7B":
            with open(file=Path(final_file_path) / Path("RWKV-4-7B.json")) as f:
                json_dict = json.load(f)
        elif file_path_or_name == "RWKV-4-14B":
            with open(file=Path(final_file_path) / Path("RWKV-4-14B.json")) as f:
                json_dict = json.load(f)

        path = None
        if json_dict !=None:
            embedding_dimension = json_dict["d_model"]
            number_of_layers = json_dict["num_decoder_layers"]
            context_length = json_dict["n_positions"]
            path = json_dict["name_or_path"]

        # if it is a url download
        
        if path == None:
            path = file_path_or_name

        if urlparse(url=path).scheme != "" and json_dict != None:
            url = path

            files = os.listdir(cache_folder_path)
            # filter by ending in .pth
            files = [f for f in files if f.endswith(".pth")]

            file = url 
            file = file.split("/")[-1]
            
            if not file in os.listdir():
                # download with wget
                file = wget.download(url=url,out=str(cache_folder_path),bar=bar_progress)
            # model file name
            file = Path(file).stem # without chkpt
            model_file_path = str(Path(cache_folder_path) / Path(file))
        else:
            model_file_path = str(Path(file_path_or_name))
            
        # configure model
        model = RWKV_RNN_Model(context_length=context_length,
                                number_of_layers=number_of_layers,
                                embedding_dim=embedding_dimension,
                                file_path=model_file_path)
        return model
