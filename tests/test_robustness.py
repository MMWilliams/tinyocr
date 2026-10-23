import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tinyocr import evaluate_robustness as evaluation
from tinyocr.robustness import DOMAINS, blanks, image_fields, rotate, select_parents


class RobustnessTests(unittest.TestCase):
    def test_parent_selection_excludes_final_and_partial_labels(self):
        receipt={'id':'r','domain':DOMAINS[0]}
        expanded=[{'id':domain,'domain':domain,'role':'dev','score_mode':'transcription'} for domain in DOMAINS[1:]]
        bad=[dict(i,id='final-'+i['id'],role='final') for i in expanded]
        bad+=[dict(i,id='partial-'+i['id'],score_mode='annotated-recall') for i in expanded]
        result=select_parents({'items':[receipt,dict(receipt,id='reserved-r',role='final')]},{'items':bad+expanded},1)
        self.assertEqual({i['id'] for i in result},{'r',*DOMAINS[1:]})
        with self.assertRaises(ValueError):select_parents({'items':[receipt]},{'items':bad},1)

    def test_rotation_geometry_and_content(self):
        im=Image.new('RGB',(30,10),'white');im.putpixel((0,0),(255,0,0))
        self.assertEqual(rotate(im,0).tobytes(),im.tobytes())
        turned=rotate(im,90)
        self.assertEqual(turned.size,(10,30))
        self.assertEqual(turned.getpixel((0,29)),(255,0,0))
        self.assertEqual(rotate(im,180).getpixel((29,9)),(255,0,0))
        self.assertGreater(rotate(im,15).height,im.height)

    def test_blanks_are_reproducible_and_distinct(self):
        a=list(blanks());b=list(blanks())
        self.assertEqual(len(a),8)
        self.assertEqual([(n,i.tobytes()) for n,i in a],[(n,i.tobytes()) for n,i in b])
        self.assertEqual(len({i.tobytes() for _,i in a}),8)

    def test_saved_image_corruption_is_not_repaired_silently(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);im=Image.new('RGB',(20,20),'white')
            record=image_fields(im,'one',root)
            self.assertEqual(image_fields(im,'one',root,False),record)
            Image.new('RGB',(20,20),'black').save(root/record['image'])
            with self.assertRaises(ValueError):image_fields(im,'one',root)

    def test_blank_false_positives_and_missing_results(self):
        ref={'id':'blank','role':'dev','domain':'procedural-negative','task':'page','score_mode':'transcription','text':''}
        rows=[{'model':name,'id':'blank','role':'dev','domain':ref['domain'],'task':'page',
               'prediction':'ABC','regions':[],'latency_ms':[1,2,3],**evaluation.score(ref,'ABC',[])} for name in evaluation.NAMES]
        evaluation.verify_rows([ref],rows)
        summary=evaluation.summaries(rows)[evaluation.NAMES[0]]['procedural-negative']
        self.assertEqual(summary['images_with_text'],1)
        self.assertEqual(summary['normalized']['blank_false_positive_chars'],3)
        self.assertIsNone(summary['normalized']['cer'])
        for bad in (rows[:1],rows+[rows[0]]):
            with self.assertRaises(ValueError):evaluation.verify_rows([ref],bad)
        bad=copy.deepcopy(rows);bad[0]['role']='final'
        with self.assertRaises(ValueError):evaluation.verify_rows([ref],bad)

    def test_completed_comparison_verifies_without_inference(self):
        with patch.object(evaluation,'REPORT') as report,patch.object(evaluation,'verify',return_value={}) as verify,patch.object(evaluation,'Baseline') as model:
            report.exists.return_value=True
            evaluation.evaluate()
            verify.assert_called_once_with()
            model.assert_not_called()
