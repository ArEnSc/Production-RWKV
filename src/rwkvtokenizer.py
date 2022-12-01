from tokenizers import Tokenizer
class RWKVTokenizer():
    @staticmethod
    def from_file(file_path:str="./RWKV-LM/RWKV-v4neo/20B_tokenizer.json"):
        return Tokenizer.from_file(file_path)