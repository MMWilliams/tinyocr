import hashlib
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tinyocr.check_expansion import check
from tinyocr.common import sha256, write_json


class IntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        write_json(self.root/'eval/manifest.json',{})
        write_json(self.root/'eval/legacy_scene_lineage.json',{'items':[]})
        items=[]
        for role,color in [('dev','white'),('final','black')]:
            path=self.root/f'{role}.png'
            im=Image.new('RGB',(10,10),color);im.save(path)
            items.append({'id':role,'role':role,'source':'screens','source_id':role,'group_id':role,
                          'image':path.name,'sha256':sha256(path),'pixel_sha256':hashlib.sha256(im.tobytes()).hexdigest(),
                          'width':10,'height':10,'score_mode':'transcription','domain':'real-screen-region','text':'Hello'})
        self.manifest={'schema':2,'purpose':'evaluation-only-expansion','sources':{'screens':{}},'items':items,
                       'pilot_manifest_sha256':sha256(self.root/'eval/manifest.json'),
                       'lineage_sha256':sha256(self.root/'eval/legacy_scene_lineage.json')}

    def run_check(self):
        path=self.root/'manifest.json';write_json(path,self.manifest)
        return check(path,self.root)

    def test_disjoint_inputs_pass(self):
        self.assertEqual(len(self.run_check()['items']),2)

    def test_same_source_image_different_pixels_rejected(self):
        self.manifest['items'][1]['source_id']='dev'
        with self.assertRaisesRegex(ValueError,'overlap'):self.run_check()

    def test_different_source_same_pixels_rejected(self):
        for k in ('image','sha256','pixel_sha256'):
            self.manifest['items'][1][k]=self.manifest['items'][0][k]
        with self.assertRaisesRegex(ValueError,'overlap'):self.run_check()

    def test_crop_cannot_cross_split(self):
        self.manifest['items'][1]['parent_id']='dev'
        with self.assertRaisesRegex(ValueError,'Crop split'):self.run_check()

    def test_partial_screen_cannot_have_full_text(self):
        item=self.manifest['items'][0]
        item.update(domain='real-screen',score_mode='annotated-recall',regions=[{'text':'Hello','box':[0,0,10,10]}])
        with self.assertRaisesRegex(ValueError,'full-page truth'):self.run_check()
