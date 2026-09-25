import unittest
from model_selection import ModelSelection

class ModelSelectionTests(unittest.TestCase):
    def test_unavailable_model_is_rejected_without_changing_selection(self):
        models=ModelSelection('open_jev','http://local/api/strategy',{'device':'cuda:0'})
        with self.assertRaises(ValueError): models.select('laya')
        self.assertEqual(models.snapshot()['selected'],'open_jev')
        self.assertFalse(models.snapshot()['options'][1]['available'])

    def test_selection_returns_new_route_and_runtime(self):
        models=ModelSelection('open_jev','http://open/api/strategy',{'model_kind':'open_jev'})
        models.routes['laya']=('http://laya/api/strategy',{'model_kind':'laya'})
        url,runtime=models.select('laya')
        self.assertEqual(url,'http://laya/api/strategy')
        self.assertEqual(runtime['model_kind'],'laya')
        self.assertEqual(models.selected,'laya')
