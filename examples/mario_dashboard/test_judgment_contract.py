import unittest
from judgment_contract import bind_plan_judgment


class PlanContractTests(unittest.TestCase):
    def setUp(self):
        self.request={'context_id':'ctx', 'plan_candidates':[{'slot':0,'id':'jump:short','skill':'transfer','action':'right_jump'}]}
        self.response={'selected_slot':0,'slot_scores':[.6,.05,.05,.05,.05,.05,.15]}

    def test_binds_real_candidate_not_untrusted_response_context(self):
        result=bind_plan_judgment({**self.response,'plan_context':'forged'},self.request)
        self.assertEqual(result['model_plan'],'jump:short')
        self.assertEqual(result['plan_context'],'ctx')
        self.assertEqual(result['plan_scores'],{'jump:short':.6})

    def test_missing_slot_does_not_become_a_plan(self):
        result=bind_plan_judgment({**self.response,'selected_slot':3},self.request)
        self.assertEqual(result['model_plan'],'')
        self.assertTrue(result['absent_slot_selected'])

    def test_defer_is_explicit_and_no_fake_goal(self):
        result=bind_plan_judgment({**self.response,'selected_slot':6},self.request)
        self.assertEqual(result['model_plan'],'')
        self.assertFalse(result['absent_slot_selected'])
        self.assertNotIn('active_goal',result)

    def test_malformed_score_or_slot_rejected(self):
        for scores in [[float('nan')]*7,[0]*7,[True]+[0]*6,[1,0],[-1,1,1,0,0,0,0]]:
            with self.assertRaises(ValueError):bind_plan_judgment({**self.response,'slot_scores':scores},self.request)
        with self.assertRaises(ValueError):bind_plan_judgment({**self.response,'selected_slot':True},self.request)

    def test_duplicate_ids_or_slot_mismatch_rejected(self):
        duplicate=[self.request['plan_candidates'][0]]*2
        with self.assertRaises(ValueError):bind_plan_judgment(self.response,{**self.request,'plan_candidates':duplicate})
        self.request['plan_candidates'][0]['slot']=1
        with self.assertRaises(ValueError):bind_plan_judgment(self.response,self.request)
