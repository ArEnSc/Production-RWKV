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
        self.model = RWKVRNN4NeoForCausalLM.from_pretrained("RWKV-4-169M")
        print(f"Model Setup:{self.model.file_name}")

    def tearDown(self):
        """Method to tear down the test fixture. Run AFTER the test methods."""
        pass
 
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

    def streaming_callback(self,s): 
        r = self.tokenizer.decode([s],skip_special_tokens=False)
        print(f"[Was Streamed a Token]:{repr(r)}")

    def test_generate_and_warmup_and_input_and_genlength_lengths(self):
        """
            1. We use nucleous sampling
            2. We warm up the context with the same thing twice
            3. We generate from that point 5 items
            expecting the length output to make be correct
        """
        context = "Hello world this is an example context."
        input_ids = self.tokenizer.encode(context).ids
        sanity_length = len(input_ids)
        print(f"Sanity Length: {sanity_length}")
        self.model.warmup_with_context(input_ids)
        
        self.assertIsNotNone(self.model.init_logits,"ensure logits got created")
        self.assertIsNotNone(self.model.init_state,"ensure initial state is created")
        gen_len = 5
        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=gen_len)
        
        total_length = len(result) 
        print(f"Result: {self.tokenizer.decode(result,skip_special_tokens=False)}")
        self.assertEqual(total_length,sanity_length + sanity_length + gen_len, "check that the length is correct")



    def test_generate_no_warmup_multiuse(self):
        # should work without a warmup
        context = "Hello world this is an example context."
        input_ids = self.tokenizer.encode(context).ids
        sanity_length = len(input_ids)
        print(f"Sanity Length: {sanity_length}")
        
        self.assertIsNone(self.model.init_logits,"ensure logits not created")
        self.assertIsNone(self.model.init_state,"ensure initial not created")
        
        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=5)
        
        total_length = len(result)
        print(f"First Run Result: {self.tokenizer.decode(result,skip_special_tokens=False)}")
        self.assertEqual(total_length,sanity_length + 5, "check that the length is correct")

        # Should not use any older states
        new_context = "This is a different context"
        input_ids = self.tokenizer.encode(new_context).ids
        sanity_length = len(input_ids)
        
        self.assertIsNone(self.model.init_logits,"ensure logits not created")
        self.assertIsNone(self.model.init_state,"ensure initial not created")

        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=1)
        
        total_length = len(result)
        print(f"Second Run Result: {self.tokenizer.decode(result,skip_special_tokens=False)}")
        self.assertEqual(total_length,sanity_length + 1, "check that the length is correct")

    def test_generate_warmup_with_empty_ids(self):
        context_one = "Hello world this is an example context. It is a great thing to have extra information here."
        input_ids = self.tokenizer.encode(context_one).ids
        sanity_length = len(input_ids)
        
        self.model.warmup_with_context(context=input_ids)
        print(f"[Check Sanity Length:] {sanity_length}")

        result = self.model.generate(input_ids=[],streaming_callback=self.streaming_callback,max_length=1,temperature=0)
        decoded = self.tokenizer.decode(result,skip_special_tokens=False)
        print(f"[decoded:]{repr(decoded)}")
        self.assertEqual(len(result),1 + len(input_ids),"Should be one token and the warmed up context")

    def test_generate_continue_state_without_warmup(self):
        
        context_one = "Hello world this is an example context."
        input_ids = self.tokenizer.encode(context_one).ids
        sanity_length = len(input_ids)
        print(f"[Check Sanity Length:] {sanity_length}")
        
        self.assertIsNone(self.model.init_logits,"ensure logits not created")
        self.assertIsNone(self.model.init_state,"ensure initial not created")
        
        self.model.update_state_after_generation(flag=True)
        gen_len = 5
        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=gen_len)
        
        total_length = len(result)
        print(f"[First Run Result:] {repr(self.tokenizer.decode(result,skip_special_tokens=False))}")
        self.assertEqual(total_length,sanity_length + gen_len, "check that the length is correct")

        self.assertIsNotNone(self.model.init_logits,"ensure logits created")
        self.assertIsNotNone(self.model.init_state,"ensure initial created")

        build_up_context_len = len(self.model.current_context)
        build_up_context = self.tokenizer.decode(self.model.current_context,skip_special_tokens=False)
        print(f"[First Run Result Current Context Build Up:] {repr(build_up_context)}")
        self.assertEqual(build_up_context_len,sanity_length + 5, "check that the build up context length is correct")
        prior_logits = self.model.init_logits.detach().clone()
        prior_state = self.model.init_state.detach().clone()
        
        context_two = "New extention to context"
        input_ids = self.tokenizer.encode(context_two).ids
        sanity_length = len(input_ids)
        
        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=gen_len)

        self.assertEqual(torch.allclose(self.model.init_logits,prior_logits),False,"ensure logits recreated")
        self.assertEqual(torch.allclose(self.model.init_state,prior_state),False,"ensure initial recreated")

        build_up_context_len_2 = len(self.model.current_context)
        build_up_context = self.tokenizer.decode(self.model.current_context,skip_special_tokens=False)
        print(f"[Second Run Result Current Context Build Up:] {repr(build_up_context)}")
        self.assertEqual(build_up_context_len_2,sanity_length + gen_len + build_up_context_len, "check that the build up context length is correct")

    # def test_base_generate_no_inputs(self):
    #     """
    #         1.This will trigger an assert because no warm up context as well as well no inputs.
    #     """
    #     with self.assertRaises(InputsNeeded):
    #         result = self.model.generate(input_ids=[],streaming_callback=self.streaming_callback,max_length=1)
    #         print("Exception Triggered InputsNeeded")

    def test_base_generate_one_no_warmup(self):
        """
            [Round 1]
            1.This test also covers input into generator, assumes no warm up generate length of 1
            should return the full context and \n.
            [Round 2]
            1.Generate again expect a different context result and \n
        """
        # no warm up 
        context_one = "Hello world this is an example context. It is a great thing to have extra information here."
        input_ids = self.tokenizer.encode(context_one).ids
        gen_len = 1

        # use greedy to get consistent 
        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=gen_len,temperature=0)
        final_result_str = self.tokenizer.decode(result,skip_special_tokens=False)
        self.assertAlmostEqual(final_result_str,"Hello world this is an example context. It is a great thing to have extra information here.\n")
        # should produce the logits continuing from given input
        token = self.tokenizer.decode([result[-1]],skip_special_tokens=False)
        print(f"[Token:] {repr(token)}")
        self.assertEqual("\n",token)


        context_one = "Hello world this is an example context. It is great."
        input_ids = self.tokenizer.encode(context_one).ids
        gen_len = 1

        # use greedy to get consistent 
        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=gen_len,temperature=0)
        final_result_str = self.tokenizer.decode(result,skip_special_tokens=False)
        self.assertAlmostEqual(final_result_str,"Hello world this is an example context. It is great.\n")
        # should produce the logits continuing from given input
        token = self.tokenizer.decode([result[-1]],skip_special_tokens=False)
        print(f"[Token:] {token}")
        self.assertEqual("\n",token)

    def test_base_generate_warmup_no_input(self):
        """
            Round 1:
            1.This test assumes warm up with context.
            2.Then generates result with warm up context and next token
            should return that statement with a \n as the last token predicted.
        """
        context_one = "Hello world this is an example context. It is a great thing to have extra information here."
        input_ids = self.tokenizer.encode(context_one).ids
        self.model.warmup_with_context(context=input_ids)
        gen_len = 1

        result = self.model.generate(input_ids=[],streaming_callback=self.streaming_callback,max_length=gen_len,temperature=0)
        final_result_string = self.tokenizer.decode(result,skip_special_tokens=False)

        self.assertEqual(final_result_string,"Hello world this is an example context. It is a great thing to have extra information here.\n","Check context makes sense.")

        token = self.tokenizer.decode([result[-1]],skip_special_tokens=False)
        print(f"[Token:] {repr(token)}")
        self.assertEqual("\n",token,"Check if the correct token is generated")

    def test_base_generate_warmup_some_input(self):
        """
            Round 1:
            1.This test assumes warm up with context.
            2.Then have some continuing the context.
            should return the warm up context and the continuing and the final token.
            Round 2:
            1.Should just use the warmup context and not save the previous input_ids
            generate with more input ids replacing a great thing to have extra information here.
        """
        context_one = "Hello world this is an example context. It is "
        input_ids = self.tokenizer.encode(context_one).ids
        self.model.warmup_with_context(context=input_ids)
        gen_len = 1

        some_input = "a great thing to have extra information here."
        input_ids = self.tokenizer.encode(some_input).ids
        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=gen_len,temperature=0)
        final_result_string = self.tokenizer.decode(result,skip_special_tokens=False)
        print(f"[Final result String:] {final_result_string}")
        self.assertEqual(final_result_string,"Hello world this is an example context. It is a great thing to have extra information here.\n","[Round1]: It should have the warm up context returned.")

        token = self.tokenizer.decode([result[-1]],skip_special_tokens=False)
        print(f"[Token:] {repr(token)}")
        self.assertEqual("\n",token,"Check if the correct token is generated")

        # Round 2
        round_2_some_input = "great."
        input_ids = self.tokenizer.encode(round_2_some_input).ids
        result = self.model.generate(input_ids=input_ids,streaming_callback=self.streaming_callback,max_length=gen_len,temperature=0)
        final_result_string = self.tokenizer.decode(result,skip_special_tokens=False)
        token = self.tokenizer.decode([result[-1]],skip_special_tokens=False)
        print(f"[Final result String:] {final_result_string}")
        self.assertEqual(final_result_string,"Hello world this is an example context. It is great.\n","[Round2]: Check if it continues properly, ignoring the input ids from the last gen.")
        print(f"[Token:] {repr(token)}")
        self.assertEqual("\n",token,"Check if the correct token is generated")


    # def test_base_generate_warmup_some_input(self):
    #     pass 

    # @unittest.skip("Demonstrating skipping")  # Skips this test only
    # @unittest.skipIf("boolean_condition", "Reason to Skip Test here.")  # Skips this test only
    # @unittest.skipUnless("boolean_condition", "Reason to Skip Test here.")  # Skips this test only
    # @unittest.expectedFailure  # This test MUST fail. If test fails, then is Ok.
    # def test_dummy(self):
    #     pass
        # self.skipTest("Just examples, use as template!.")  # Skips this test only
        # self.assertEqual(a, b)  # a == b
        # self.assertNotEqual(a, b)  # a != b
        # self.assertTrue(x)  # bool(x) is True
        # self.assertFalse(x)  # bool(x) is False
        # self.assertIs(a, b)  # a is b
        # self.assertIsNot(a, b)  # a is not b
        # self.assertIsNone(x)  # x is None
        # self.assertIsNotNone(x)  # x is not None
        # self.assertIn(a, b)  # a in b
        # self.assertNotIn(a, b)  # a not in b
        # self.assertIsInstance(a, b)  # isinstance(a, b)
        # self.assertNotIsInstance(a, b)  # not isinstance(a, b)
        # self.assertAlmostEqual(a, b)  # round(a-b, 7) == 0
        # self.assertNotAlmostEqual(a, b)  # round(a-b, 7) != 0
        # self.assertGreater(a, b)  # a > b
        # self.assertGreaterEqual(a, b)  # a >= b
        # self.assertLess(a, b)  # a < b
        # self.assertLessEqual(a, b)  # a <= b
        # self.assertRegex(s, r)  # r.search(s)
        # self.assertNotRegex(s, r)  # not r.search(s)
        # self.assertItemsEqual(a, b)  # sorted(a) == sorted(b) and works with unhashable objs
        # self.assertDictContainsSubset(a, b)  # all the key/value pairs in a exist in b
        # self.assertCountEqual(a, b)  # a and b have the same elements in the same number, regardless of their order
        # # Compare different types of objects
        # self.assertMultiLineEqual(a, b)  # Compare strings
        # self.assertSequenceEqual(a, b)  # Compare sequences
        # self.assertListEqual(a, b)  # Compare lists
        # self.assertTupleEqual(a, b)  # Compare tuples
        # self.assertSetEqual(a, b)  # Compare sets
        # self.assertDictEqual(a, b)  # Compare dicts
        # # To Test code that MUST Raise Exceptions:
        # self.assertRaises(SomeException, callable, *args, **kwds)  # callable Must raise SomeException
        # with self.assertRaises(SomeException) as cm:
        #     do_something_that_raises() # This line  Must raise SomeException
        # # To Test code that MUST Raise Warnings (see std lib warning module):
        # self.assertWarns(SomeWarning, callable, *args, **kwds)  # callable Must raise SomeWarning
        # with self.assertWarns(SomeWarning) as cm:
        #     do_something_that_warns() # This line  Must raise SomeWarning
        # # Assert messages on a Logger log object.
        # self.assertLogs(logger, level)
        # with self.assertLogs('foo', level='INFO') as cm:
        #     logging.getLogger('foo').info('example message')  # cm.output is 'example message'


if __name__.__contains__("__main__"):
    unittest.main()
    # Run just 1 test.
    # unittest.main(defaultTest='TestFoo.test_foo', warnings='ignore')