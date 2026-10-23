"""Run local CPU OCR on one image using a measured baseline/candidate."""
import argparse
import sys
from pathlib import Path

from .baseline import Baseline, MODEL_NAMES
from .common import write_json


def main():
    # Windows redirected stdout may default to cp1252; OCR can emit any Unicode.
    if hasattr(sys.stdout,'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image',type=Path)
    parser.add_argument('--model',choices=MODEL_NAMES,default='v6-w8-rec')
    parser.add_argument('--output',type=Path,help='Optional JSON destination (text and detected regions)')
    args=parser.parse_args()
    if not args.image.is_file():
        parser.error('image must be an existing local file')
    text,regions=Baseline(args.model).predict(args.image,'page')
    if args.output:
        if args.output.resolve()==args.image.resolve():
            parser.error('output must not overwrite the source image')
        write_json(args.output,{'model':args.model,'image':str(args.image.resolve()),'text':text,'regions':regions})
    print(text)


if __name__=='__main__':
    main()
