import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tinyocr import evaluate_final as final


def item(identifier='f'):
    return {'id':identifier,'role':'final','domain':'full-scene','task':'page',
            'score_mode':'transcription','text':'cat'}


def row(ref,name,text='cat',regions=None):
    regions=regions or []
    return {'model':name,**{k:ref[k] for k in ('id','role','domain','task','score_mode')},
            'prediction':text,'regions':regions,'latency_ms':[1.,2.,3.],
            **final.score(ref,text,regions)}


class FinalTests(unittest.TestCase):
    def test_select_only_final_and_reject_bad_partition(self):
        f=item()
        self.assertEqual(final.select_final({'items':[dict(f,role='dev',id='d'),f]}),[f])
        for items in ([],[dict(f,role='dev')],[f,f]):
            with self.assertRaises(ValueError):final.select_final({'items':items})

    def test_partial_screen_has_recall_without_cer(self):
        refs=[{'text':'Save','box':[0,0,100,20]}]
        result=final.score({'score_mode':'partial-regions','regions':refs},'Save extra',refs)
        self.assertEqual(result,{'annotation_recovery':{'annotated':1,'recovered':1}})

    def test_character_regression_fails_transfer(self):
        f=item()
        result=final.compare([f],{final.NAMES[0]:[row(f,final.NAMES[0])],
                                final.NAMES[1]:[row(f,final.NAMES[1],'cut')]})
        self.assertFalse(result['transfer_pass'])
        self.assertEqual(result['transfer_checks'][0]['baseline'],0)
        self.assertEqual(result['transfer_checks'][0]['candidate'],1)

    def test_lost_label_fails_even_when_total_recall_improves(self):
        regions=[{'text':text,'box':[0,y,100,y+20]} for text,y in [('Save',0),('Cancel',30),('Help',60)]]
        f=dict(item(),domain='real-screen',score_mode='partial-regions',regions=regions)
        result=final.compare([f],{final.NAMES[0]:[row(f,final.NAMES[0],regions=regions[:1])],
                                final.NAMES[1]:[row(f,final.NAMES[1],regions=regions[1:])]})
        self.assertFalse(result['transfer_pass'])
        self.assertEqual(result['transfer_checks'][0]['lost'],[{'id':'f','region':0,'reference':'Save'}])

    def test_coverage_and_metric_corruption_rejected(self):
        f=item();rows=[row(f,name) for name in final.NAMES]
        final.verify_rows([f],rows)
        variants=[rows[:1],rows+[rows[0]]]
        for field,value in [('id','dev-id'),('role','dev'),('latency_ms',[1,2,float('nan')])]:
            bad=copy.deepcopy(rows);bad[0][field]=value;variants.append(bad)
        bad=copy.deepcopy(rows);bad[0]['normalized']['char_errors']=4;variants.append(bad)
        for bad in variants:
            with self.assertRaises(ValueError):final.verify_rows([f],bad)

    def test_duplicate_or_torn_journal_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'journal.jsonl'
            for data in ('{"model":"a","id":"f"}\n'*2,'{"model":'):
                path.write_text(data,encoding='utf-8')
                with self.assertRaises(ValueError):final.read_journal(path)

    def test_completed_evaluation_does_not_infer(self):
        with patch.object(final,'REPORT') as report,patch.object(final,'verify',return_value={'saved':True}) as verify,patch.object(final,'Baseline') as model:
            report.exists.return_value=True
            self.assertEqual(final.evaluate(),{'saved':True})
            verify.assert_called_once_with()
            model.assert_not_called()
