
import unittest

from prwkv.rwkvtokenizer import RWKVTokenizer
from prwkv.rwkvrnnmodel import RWKVRNN4NeoForCausalLM
from prwkv.rwkvrnnmodel import InputsNeeded
from pathlib import Path
import torch
class GenerateTests(unittest.TestCase):

    def setUp(self):
        """Method to prepare the test fixture. Run BEFORE the test methods."""
        self.tokenizer = RWKVTokenizer.default()
        # This test is valid for the following file Lastest model using half https://huggingface.co/BlinkDL/rwkv-4-pile-169m/resolve/main/RWKV-4-Pile-169M-20220807-8023.pth
        #
        self.model = RWKVRNN4NeoForCausalLM.from_pretrained("RWKV-4-169M")
        self.model.half()
        print(f"Model Setup:{self.model.file_name}")
        self.context = "\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese."
        self.context_ids = self.tokenizer.encode(self.context).ids
        self.context_length = len(self.context_ids)
        self.test_generation_with_string = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.\nThe researchers found that the dragons were able to communicate with each other, and that they were able to communicate with each other.\n\nThe researchers also found that the dragons were able to communicate with each other, and that they were able to communicate with each other.\n\nThe researchers also found that the dragons were able to communicate with each other, and that they were able to communicate with each other.\n\nThe researchers also found that the dragons were able to communicate with each other, and that they were able to communicate with each other.\n\nThe researchers also found that the dragons were able to communicate with each other, and'
        self.test_generation_with_string_ids = self.tokenizer.encode(self.test_generation_with_string).ids

    def tearDown(self):
        """Method to tear down the test fixture. Run AFTER the test methods."""
        pass

    def streaming_callback(self,token):
        print(f"Processing {repr(self.tokenizer.decode([token],skip_special_tokens=False))}")


    def test_no_input_no_warmup_with_context(self):
        """
        Base Case No Inputs No Warm Up With Context means Inputs Needed Assertion.
        """
        with self.assertRaises(InputsNeeded):
            input_ids = []
            self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0)

    def test_warmup_with_context_length_0(self):
        """
        Warm up context and generate two tokens with Empty input_ids. No State Resuming.
        """
        input_ids = []
        self.model.should_return_full_context(flag=True)
        self.model.warmup_with_context(self.context_ids)
        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=0,streaming_callback=self.streaming_callback)

        # check that context is returned
        context = self.tokenizer.decode(generation,skip_special_tokens=False)
        self.assertEqual(context,self.context,"Check that the context is the same")

         # take the last token of the context "".""
        result = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        self.assertEqual(result,".","Generation at 0 length") #chinese. 

        self.assertEqual(self.context_length , len(generation),"Should be the length of the context + 0")

    def test_warmup_with_context_length_1(self):
        """
        Warm up context and generate two tokens with Empty input_ids. No State Resuming.
        """
        input_ids = []
        self.model.should_return_full_context(flag=True)

        self.model.warmup_with_context(self.context_ids)

        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=1,streaming_callback=self.streaming_callback)

        # check that context is returned
        context = self.tokenizer.decode(generation[:self.context_length],skip_special_tokens=False)
        self.assertEqual(context,self.context,"Check that the context is the same")

        # take the first token of generation "."
        result = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        self.assertEqual(result,"\n","Generation at 1 length")

        self.assertEqual(self.context_length + 1, len(generation),"Should be the length of the context + 1")


    def test_warmup_with_context_length_2(self):
        """
        Warm up context and generate two tokens with Empty input_ids. No State Resuming.
        """
        input_ids = []
        self.model.should_return_full_context(flag=True)

        self.model.warmup_with_context(self.context_ids)
        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=2,streaming_callback=self.streaming_callback) # take the second token

        # check that context is returned
        context = self.tokenizer.decode(generation[:self.context_length],skip_special_tokens=False)
        self.assertEqual(context,self.context,"Check that the context is the same")

        result = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        self.assertEqual(result,"The","Generation at 2 length")

        self.assertEqual(self.context_length + 2, len(generation),"Should be the length of the context + 2")

    def test_no_warmup_with_input_ids_as_context_length_2(self):
        """
        No Warm up context and generate 0 tokens with short input_ids as context.
        """
        input_ids = self.context_ids
        self.model.should_return_full_context(flag=False)
        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=2,streaming_callback=self.streaming_callback) # take the second token
        print(repr(self.tokenizer.decode(generation,skip_special_tokens=False)))
        last_token = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        self.assertEqual(last_token,"\n","Make sure the second token is generated.")

    def test_warmup_with_state_resume_one_genrations(self):
        """
        Use Warm up context and generate 3 tokens and continue generation with state.
        First phase just generate the same 3 tokens twice.
        Second phase save the state and generate the next 3 tokens. 
        """
        print("First Phase")

        input_ids = []
        self.model.update_state_after_generation(flag=False)

        self.model.warmup_with_context(self.context_ids)
        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=3,streaming_callback=self.streaming_callback)

        last_token = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        second_last_token = self.tokenizer.decode([generation[-2]],skip_special_tokens=False)
        third_last_token = self.tokenizer.decode([generation[-3]],skip_special_tokens=False)
        
        self.assertEqual(" researchers",last_token,"check the last token")
        self.assertEqual("The",second_last_token,"check the second last")
        self.assertEqual("\n",third_last_token,"check the third lasst token")
    

        # continue generation with the without stored state
        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=3,streaming_callback=self.streaming_callback)

        last_token = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        second_last_token = self.tokenizer.decode([generation[-2]],skip_special_tokens=False)
        third_last_token = self.tokenizer.decode([generation[-3]],skip_special_tokens=False)

        self.assertEqual(" researchers",last_token,"check the last token")
        self.assertEqual("The",second_last_token,"check the second last")
        self.assertEqual("\n",third_last_token,"check the third lasst token")
        print("Done First Phase")

        print("Second Phase")
        self.model.update_state_after_generation(flag=True)
         # continue generation with the without stored state
        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=3,streaming_callback=self.streaming_callback)

        last_token = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        second_last_token = self.tokenizer.decode([generation[-2]],skip_special_tokens=False)
        third_last_token = self.tokenizer.decode([generation[-3]],skip_special_tokens=False)
        
        self.assertEqual(" researchers",last_token,"check the last token")
        self.assertEqual("The",second_last_token,"check the second last")
        self.assertEqual("\n",third_last_token,"check the third lasst token")

        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=3,streaming_callback=self.streaming_callback)

        last_token = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        second_last_token = self.tokenizer.decode([generation[-2]],skip_special_tokens=False)
        third_last_token = self.tokenizer.decode([generation[-3]],skip_special_tokens=False)

        self.assertEqual(" the",last_token,"check the last token")
        self.assertEqual(" that",second_last_token,"check the second last")
        self.assertEqual(" found",third_last_token,"check the third last token")

        generation = self.model.generate(input_ids=input_ids,temperature=0,repetition_penalty=0,max_length=1,streaming_callback=self.streaming_callback)

        last_token = self.tokenizer.decode([generation[-1]],skip_special_tokens=False)
        self.assertEqual(" dragons",last_token,"check the last token")
        # implied that the context is returned :)

    def test_load_cpu(self):
        self.model.cpu()
        self.assertEqual(self.model.args.FLOAT_MODE,"fp32","Make sure flag is floating point 32")
        self.assertEqual(self.model.args.RUN_DEVICE,"cpu"," Using CPU")
 
    def test_context_save_load(self):
        context = "Hello world this is an example context."
        input_ids = self.tokenizer.encode(context).ids

        self.model.warmup_with_context(input_ids)

        self.assertAlmostEqual(self.model.warmup_context,self.model.warmup_context,"check that the context is stored.")
        self.model.save_context(save_path_and_name=Path("./test_save"),context_decoded=context)

        prev_logits = self.model.init_logits.detach().clone()
        prev_state = self.model.init_state.detach().clone()

        context_string, model_name = self.model.load_context(load_path=Path("./test_save"))
       
        self.assertTrue(torch.allclose(prev_logits,self.model.init_logits), "check prior logits")
        self.assertTrue(torch.allclose(prev_state,self.model.init_state),"check prior state")

        self.assertEqual(context,context_string,"check if the og context matches up.")
        self.assertEqual(model_name,self.model.file_name,"check model name")

    def test_fullcontext(self):
        """
        This test checks if the full context will be returned or not.
        """
        self.model.update_state_after_generation(True)
        self.model.should_return_full_context(True)
        # answer to the full context being returned
        result1 = '\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.\nThe'

        self.model.warmup_with_context(self.context_ids)
        generation = self.model.generate(input_ids=[],streaming_callback=None,max_length=2,repetition_penalty=0,temperature=0,stop_on_eos=True)
        self.assertEqual(len(generation), self.context_length+2, 'check if the context returned is full context length or just the generated context length.')
        test1 = self.tokenizer.decode(generation,skip_special_tokens=False)
        self.assertEqual(test1,result1,"Should have the following output with the whole context and the generated words")

        self.model.should_return_full_context(False)
        generation = self.model.generate(input_ids=[],streaming_callback=None,max_length=2,repetition_penalty=0,temperature=0,stop_on_eos=True)
        self.assertEqual(len(generation), 2, 'check if the context returned is full context length or just the generated context length.')
        test2 = self.tokenizer.decode(generation,skip_special_tokens=False)
        self.assertEqual(test2,' researchers found',"should just generate length 2 and the remaining string")

        self.model.should_return_full_context(False)
        generation = self.model.generate(input_ids=[],streaming_callback=None,max_length=2,repetition_penalty=0,temperature=0,stop_on_eos=True)
        self.assertEqual(len(generation), 2, 'check if the context returned is full context length or just the generated context length.')
        test3 = self.tokenizer.decode(generation,skip_special_tokens=False)
        self.assertEqual(test3,' that the')

        # keeps that context state as a buffer but doesn't use it to generate 
        self.model.should_return_full_context(True)
        generation = self.model.generate(input_ids=[],streaming_callback=None,max_length=1,repetition_penalty=0,temperature=0,stop_on_eos=True)
        test4 = self.tokenizer.decode(generation,skip_special_tokens=False)
        self.assertEqual(test4,'\nIn a shocking finding, scientist discovered a herd of dragons living in a remote, previously unexplored valley, in Tibet. Even more surprising to the researchers was the fact that the dragons spoke perfect Chinese.\nThe researchers found that the dragons', "should return the full context again with the extra generated stuff prior")

    def test_context_start_0_input_id_with_context_no_update(self):
        # no update should just generate \n on greedy for that specific prompt
        self.model.update_state_after_generation(False)
        self.model.should_return_full_context(True)
        # it's generating an extra token? 
        generation = self.model.generate(input_ids=self.context_ids,streaming_callback=self.streaming_callback,max_length=0,repetition_penalty=0,temperature=0,stop_on_eos=True)        
        test1 = self.tokenizer.decode(generation,skip_special_tokens=False)

        print(repr(test1))
        print(repr(self.context))
        
        self.assertEqual(test1,self.context,"Should just return the full context nothing should be generated time step 1 and didn't do a warm up")

        self.model.update_state_after_generation(False)
        self.model.should_return_full_context(False)
        # it's generating an extra token? 
        generation = self.model.generate(input_ids=self.context_ids,streaming_callback=self.streaming_callback,max_length=0,repetition_penalty=0,temperature=0,stop_on_eos=True)        
        test1 = self.tokenizer.decode(generation,skip_special_tokens=False)
        print(repr(test1))
        print(repr(self.context))

        self.assertEqual(test1,self.context,"Should return the full context because we have generated something prior from time step 1.")

        self.model.should_return_full_context(True)
        generation = self.model.generate(input_ids=self.context_ids,streaming_callback=self.streaming_callback,max_length=0,repetition_penalty=0,temperature=0,stop_on_eos=True)        
        test1 = self.tokenizer.decode(generation,skip_special_tokens=False)
        print(repr(test1))
        print(repr(self.context))

        self.assertEqual(test1,self.context+self.context+self.context,"Should return the full context because we have generated something prior from time step 1 2 and this round number 3")

    def test_context_start_1_input_id_with_context_no_update(self):
        # no update should just generate \n on greedy for that specific prompt
        self.model.update_state_after_generation(False)
        self.model.should_return_full_context(True)
        # it's generating an extra token? 
        generation = self.model.generate(input_ids=self.context_ids,streaming_callback=self.streaming_callback,max_length=1,repetition_penalty=0,temperature=0,stop_on_eos=True)        
        test1 = self.tokenizer.decode(generation,skip_special_tokens=False)

        print(repr(test1))
        print(repr(self.context))

        self.assertEqual(test1,self.context+"\n","Should just return the full context +\n nothing should be generated time step 1 and didn't do a warm up")


    def test_context_start_2_input_id_with_context_no_update(self):
        # no update should just generate \n on greedy for that specific prompt
        self.model.update_state_after_generation(False)
        self.model.should_return_full_context(True)
        # it's generating an extra token? 
        generation = self.model.generate(input_ids=self.context_ids,streaming_callback=self.streaming_callback,max_length=1,repetition_penalty=0,temperature=0,stop_on_eos=True)        
        test1 = self.tokenizer.decode(generation,skip_special_tokens=False)
if __name__.__contains__("__main__"):
    unittest.main()