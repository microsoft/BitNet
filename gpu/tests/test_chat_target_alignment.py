import sys
from pathlib import Path
import unittest
sys.path.insert(0,str(Path(__file__).parents[1]))
from tokenizer import Tokenizer,ChatFormat

class TargetAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.format = ChatFormat(Tokenizer(str(Path(__file__).parents[1]/"tokenizer.model")))

    def test_completion_header_has_masked_targets(self):
        dialog=[{"role":"user","content":"Hello"}]
        tokens,targets=self.format.encode_dialog_prompt(dialog,completion=True,return_target=True)
        header=self.format.encode_header({"role":"assistant","content":""})
        self.assertEqual(len(tokens),len(targets))
        self.assertEqual(tokens[-len(header):],header)
        self.assertEqual(targets[-len(header):],[-1]*len(header))

    def test_existing_assistant_targets_are_preserved(self):
        dialog=[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]
        tokens,targets=self.format.encode_dialog_prompt(dialog,completion=True,return_target=True)
        base_tokens,base_targets=self.format.encode_dialog_prompt(dialog,completion=False,return_target=True)
        self.assertEqual(tokens[:len(base_tokens)],base_tokens)
        self.assertEqual(targets[:len(base_targets)],base_targets)
        self.assertEqual(len(tokens),len(targets))

    def test_empty_dialog_completion_and_tokens_only_match(self):
        tokens,targets=self.format.encode_dialog_prompt([],completion=True,return_target=True)
        self.assertEqual(len(tokens),len(targets))
        self.assertEqual(targets,[-1]*len(tokens))
        self.assertEqual(tokens,self.format.encode_dialog_prompt([],completion=True))

if __name__=="__main__":unittest.main()
