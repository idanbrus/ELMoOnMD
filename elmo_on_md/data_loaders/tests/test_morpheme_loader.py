from unittest import TestCase

from elmo_on_md.data_loaders.tree_bank_loader import Morphemes_loader
import numpy as np


class TestMorphemes_loader(TestCase):
    def test_load_data(self):
        morpheme_loader = Morphemes_loader()
        data = morpheme_loader.load_data()
        self.assertGreater(len(data['train']), 0)
        self.assertEqual(len(data['train'][0]),5) # 5 words
        self.assertEqual(len(data['train'][0][0]),51) #51 morphemes
        self.assertEqual(data['train'][0][0].sum(),1) #first word is quotes
    def test__map_pos(self):
        morpheme_loader = Morphemes_loader()
        self.assertEqual(morpheme_loader._map_pos('P1'),0)
        self.assertEqual(morpheme_loader._map_pos('P1'), 0)
        self.assertEqual(morpheme_loader._map_pos('P2'), 1)
        self.assertEqual(morpheme_loader._map_pos('P1'), 0)
        self.assertEqual(morpheme_loader._map_pos('P3'), 2)

    def test__get_pos_and_token_id(self):
        morpheme_loader = Morphemes_loader()
        self.assertEqual(morpheme_loader._get_pos_and_token_id('3	4	ו	ו	CONJ	CONJ	_	4'), ('CONJ',4))
        self.assertEqual(morpheme_loader._get_pos_and_token_id('4	5	בגדול	בגדול	RB	RB	_	4'),('RB',4))
        self.assertEqual(morpheme_loader._get_pos_and_token_id('1	2	תהיה	היה	COP	COP	gen=F|num=S|per=3	2'),('COP',2))

    def test__set_to_vec(self):
        morpheme_loader = Morphemes_loader()
        morpheme_loader.max_pos_id=2
        self.assertEqual(list(morpheme_loader._set_to_vec({0,1})),[1,1])
        self.assertEqual(list(morpheme_loader._set_to_vec({})),[0,0])
        self.assertEqual(list(morpheme_loader._set_to_vec({0})),[1,0])
        morpheme_loader.max_pos_id = 3
        self.assertEqual(len(morpheme_loader._set_to_vec({})), 3)

    def test__get_sentence_morpheme_map(self):
        morpheme_loader = Morphemes_loader()
        test_string = """0	1	"	_	yyQUOT	yyQUOT	_	1
1	2	תהיה	היה	COP	COP	gen=F|num=S|per=3	2
2	3	נקמה	נקמה	NN	NN	gen=F|num=S	3
3	4	ו	ו	CONJ	CONJ	_	4
4	5	בגדול	בגדול	RB	RB	_	4
5	6	.	_	yyDOT	yyDOT	_	5"""
        test_answer = morpheme_loader._get_sentence_morpheme_map(test_string)
        self.assertEqual(test_answer[0],(set([0])))
        self.assertEqual(test_answer[3],(set([3,4])))

