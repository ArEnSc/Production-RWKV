from tokenizers import Tokenizer
from pathlib import Path

class PathForTokenizerInvalid(Exception):
    "Need Path for Tokenizer."
    pass

class RWKVTokenizer():
    
    @staticmethod
    def default():
        file_path = Path(__file__)
        final_file_path = Path(file_path.parent) / Path("data") / Path("20B_tokenizer.json")
        path = str(final_file_path)
        return Tokenizer.from_file(path=path)

    @staticmethod
    def from_file(tokenizer_file_path:str=None):
        if tokenizer_file_path != None:
            return Tokenizer.from_file(tokenizer_file_path)
        else:
            raise PathForTokenizerInvalid
