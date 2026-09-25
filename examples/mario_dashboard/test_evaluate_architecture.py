import unittest
from evaluate_architecture import compare


class PairedEvaluationTests(unittest.TestCase):
    def test_progress_never_hides_lost_clear_or_kills(self):
        a={'flag_get':True,'max_x':400,'objectives':{'confirmed_eliminations':3,'score':100}}
        b={'flag_get':False,'max_x':500,'objectives':{'confirmed_eliminations':2,'score':110}}
        result=compare(a,b,'hunter')
        self.assertEqual(result['max_x_delta'],100)
        self.assertEqual(result['regression_signals'],['lost_stage_clear','fewer_confirmed_eliminations'])
