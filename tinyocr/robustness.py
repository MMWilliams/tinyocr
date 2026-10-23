"""Deterministic development-only rotation and procedural-negative diagnostics."""
import argparse
import hashlib
import json
import random

from PIL import Image, ImageDraw

from .common import MANIFEST, ROOT, sha256, write_json
from .expand import EXPANSION

PATH = ROOT/'eval/robustness-v1.json'
ANGLES = (0, -15, 15, 90, 180, 270)
DOMAINS = ('receipt-page', 'full-scene', 'real-screen-region')


def ranked(items):
    return sorted(items, key=lambda i: hashlib.sha256(
        ('tinyocr-robustness-v1:'+i['id']).encode()).hexdigest())


def select_parents(pilot, expansion, count=4):
    if count < 1:
        raise ValueError('Parent count must be positive')
    eligible = [i for i in pilot['items'] if i['domain']=='receipt-page' and i.get('role','dev')=='dev']
    eligible += [i for i in expansion['items'] if i['role']=='dev' and
                 i['domain'] in DOMAINS[1:] and i['score_mode']=='transcription']
    selected=[]
    for domain in DOMAINS:
        rows=ranked([i for i in eligible if i['domain']==domain])[:count]
        if len(rows)!=count:
            raise ValueError('Insufficient development parents: '+domain)
        selected.extend(rows)
    if len({i['id'] for i in selected})!=len(selected):
        raise ValueError('Duplicate parent IDs')
    return selected


def rotate(image, angle):
    if angle not in ANGLES:
        raise ValueError('Undeclared rotation')
    return image.convert('RGB').rotate(angle, resample=Image.Resampling.BICUBIC,
                                       expand=True, fillcolor='white')


def blanks():
    size=(640,480)
    for name,color in [('white','white'),('black','black'),('gray','#808080')]:
        yield name,Image.new('RGB',size,color)
    gradient=Image.new('RGB',size)
    draw=ImageDraw.Draw(gradient)
    for y in range(size[1]):
        v=32+int(192*y/(size[1]-1))
        draw.line((0,y,size[0],y),fill=(v,v,v))
    yield 'gradient',gradient
    for name in ('checker','lines','circles'):
        im=Image.new('RGB',size,'#dddddd');draw=ImageDraw.Draw(im)
        if name=='checker':
            for y in range(0,size[1],40):
                for x in range(0,size[0],40):
                    if (x//40+y//40)%2:draw.rectangle((x,y,x+39,y+39),fill='#aaaaaa')
        elif name=='lines':
            for y in range(15,size[1],25):draw.line((0,y,size[0],y),fill='#777777',width=2)
        else:
            for x,y,r in ((90,90,35),(300,210,60),(500,360,90)):
                draw.ellipse((x-r,y-r,x+r,y+r),outline='#999999',width=4)
        yield name,im
    yield 'noise',Image.frombytes('RGB',size,random.Random(20261023).randbytes(size[0]*size[1]*3))


def image_fields(image, identifier, root=ROOT, restore=True):
    path=root/'data/robustness/images'/f'{identifier}.png'
    if not path.exists():
        if not restore:raise ValueError('Missing diagnostic image')
        path.parent.mkdir(parents=True,exist_ok=True)
        image.save(path,format='PNG')
    with Image.open(path) as saved:
        if saved.size!=image.size or saved.convert('RGB').tobytes()!=image.tobytes():
            raise ValueError('Diagnostic image differs from declared transformation')
    return {'image':path.relative_to(root).as_posix(),'sha256':sha256(path),
            'width':image.width,'height':image.height,
            'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest()}


def prepare(restore=True):
    pilot=json.loads(MANIFEST.read_text(encoding='utf-8'))
    expansion=json.loads(EXPANSION.read_text(encoding='utf-8'))
    parents=select_parents(pilot,expansion)
    items=[]
    for parent in parents:
        path=ROOT/parent['image']
        if sha256(path)!=parent['sha256']:
            raise ValueError('Parent image changed')
        with Image.open(path) as opened:
            original=opened.convert('RGB')
        for angle in ANGLES:
            identifier=f"rotation-{parent['id']}-{angle}"
            items.append({'id':identifier,'role':'dev','parent_id':parent['id'],
                          'parent_image_sha256':parent['sha256'],'source':parent['source'],
                          'source_id':parent['source_id'],'parent_domain':parent['domain'],
                          'domain':f"{parent['domain']}/rotation-{angle}",'task':'page',
                          'angle_degrees_ccw':angle,'reference_kind':'inherited-development',
                          'score_mode':'transcription','text':parent['text'],
                          **image_fields(rotate(original,angle),identifier,restore=restore)})
    for name,image in blanks():
        identifier='procedural-negative-'+name
        items.append({'id':identifier,'role':'dev','source':'local-procedural-negative',
                      'source_id':name,'domain':'procedural-negative','task':'page',
                      'score_mode':'transcription','reference_kind':'authored-no-text','text':'',
                      **image_fields(image,identifier,restore=restore)})
    manifest={'schema':1,'purpose':'development-robustness-diagnostics',
              'pilot_manifest_sha256':sha256(MANIFEST),'expanded_manifest_sha256':sha256(EXPANSION),
              'selection':'4 parents/domain by SHA256 tinyocr-robustness-v1:id; development only',
              'angles_ccw':list(ANGLES),'parent_input_count':len(parents),
              'original_source_parent_count':len({(p['source'],p['source_id']) for p in parents}),
              'items':items,'limitations':[
                  'Derivatives are correlated development inputs, not independent or final data',
                  'Rotated full-page CER also measures failure to restore original reading order',
                  'Procedural backgrounds are not representative real-world text-free photographs',
                  'No new real receipt photos, difficult signs or fully transcribed screens collected']}
    if PATH.exists():
        if json.loads(PATH.read_text(encoding='utf-8'))!=manifest:
            raise ValueError('Frozen robustness manifest differs from regeneration')
    elif restore:
        write_json(PATH,manifest)
    else:
        raise ValueError('Missing robustness manifest')
    print(f'Robustness verified: {len(items)} dev items, {len(parents)} existing inputs, zero final parents')
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=('prepare','verify'))
    args=parser.parse_args()
    prepare(restore=args.action=='prepare')
