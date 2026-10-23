import io
import unittest
import zipfile

from tinyocr.check_expansion import check_training_overlap, select_dev
from tinyocr.remotezip import RangeReader, read_member
from tinyocr.screen_labels import annotated_recall, parse_screen


class ScreenTests(unittest.TestCase):
    def test_parse_nested_labels_and_ignore_descriptions(self):
        label = 'PICTOGRAM a save icon 0 100 0 100, BUTTON Save 100 300 100 200 (TEXT Save 110 290 110 190), TEXT Total: $12.50 100 400 300 400'
        regions = parse_screen(label, 500, 1000)
        self.assertEqual([r['text'] for r in regions], ['Save','Total: $12.50'])
        self.assertEqual(regions[0]['box'], [50,100,150,200])

    def test_commas_and_digits_remain_text(self):
        regions = parse_screen('TEXT Mountain View, CA 10 900 100 200, TEXT 42 10 100 200 300',1000,1000)
        self.assertEqual([r['text'] for r in regions], ['Mountain View, CA','42'])

    def test_partial_label_recall_and_location(self):
        refs = [{'text':'Save','box':[0,0,100,20]}, {'text':'Cancel','box':[0,30,100,50]}]
        predictions = [{'text':'Save changes','box':[0,0,110,20]}, {'text':'extra unannotated text','box':[0,80,100,100]}]
        self.assertEqual(annotated_recall(refs,predictions), {'recovered':1,'annotated':2})
        self.assertEqual(annotated_recall(refs,[]), {'recovered':0,'annotated':2})
        self.assertEqual(annotated_recall(refs,[{'text':'Save','box':[200,0,300,20]}])['recovered'],0)
        self.assertEqual(annotated_recall(refs,[{'text':'Saved','box':[0,0,100,20]}])['recovered'],0)

    def test_multiline_label(self):
        ref=[{'text':'hello world','box':[0,0,100,60]}]
        pred=[{'text':'hello','box':[0,0,80,20]},{'text':'world','box':[0,30,80,50]}]
        self.assertEqual(annotated_recall(ref,pred)['recovered'],1)


class SplitTests(unittest.TestCase):
    def test_final_is_never_selected(self):
        self.assertEqual(select_dev({'items':[{'id':'d','role':'dev'},{'id':'f','role':'final'}]}),[{'id':'d','role':'dev'}])

    def test_source_or_pixel_overlap_rejected(self):
        evaluation = [{'source':'textocr','source_id':'a','pixel_sha256':'same'}]
        for train in [{'source':'textocr','source_id':'a','source_split':'train','pixel_sha256':'different'},
                      {'source':'elsewhere','source_id':'b','source_split':'train','pixel_sha256':'same'},
                      {'source':'receipts','source_id':'new','source_split':'test','pixel_sha256':'new'},
                      {'source':'textocr','source_id':'new','source_split':'val','pixel_sha256':'new'}]:
            with self.assertRaises(ValueError):
                check_training_overlap([train], evaluation)
        with self.assertRaises(ValueError):
            check_training_overlap([{'source':'unknown'}], evaluation)


class Response:
    def __init__(self, payload, spec, status=206, wrong=False):
        a,b=spec.removeprefix('bytes=').split('-')
        start=max(0,len(payload)-int(b)) if not a else int(a)
        end=len(payload)-1 if not a else min(int(b),len(payload)-1)
        self.body=payload[start:end+1]
        self.status_code=status
        self.headers={'Content-Range':f'bytes {start+int(wrong)}-{end}/{len(payload)}'}
    def __enter__(self): return self
    def __exit__(self,*args): return False
    def iter_content(self,n): yield self.body


class Session:
    def __init__(self,payload,**kwargs): self.payload,self.kwargs=payload,kwargs
    def get(self,url,**kwargs): return Response(self.payload,kwargs['headers']['Range'],**self.kwargs)


class RangeTests(unittest.TestCase):
    def test_random_access_zip(self):
        buf=io.BytesIO()
        with zipfile.ZipFile(buf,'w') as z:
            z.writestr('unused',b'x'*100000)
            z.writestr('image.jpg',b'expected image bytes')
        reader=RangeReader('test',Session(buf.getvalue()))
        with zipfile.ZipFile(reader) as z:
            self.assertEqual(read_member(z,'image.jpg'),b'expected image bytes')
        self.assertLess(reader.bytes_downloaded,len(buf.getvalue()))

    def test_refuse_unbounded_response(self):
        with self.assertRaisesRegex(ValueError,'refusing full download'):
            RangeReader('test',Session(b'hello',status=200))

    def test_wrong_range_rejected(self):
        with self.assertRaisesRegex(ValueError,'wrong byte range'):
            RangeReader('test',Session(b'hello',wrong=True))
