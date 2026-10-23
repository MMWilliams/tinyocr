# Next independent OCR collection

The previous final set is consumed. Keep its evidence and candidate lock intact. This plan specifies the next collection before model selection; **no fresh final partition is yet frozen**. Rotation derivatives and procedural blanks in the separate robustness suite are development controls and do not count toward the collection below.

## Acquisition and source review, 2026-10-23

| Candidate source | Primary evidence | Decision |
| --- | --- | --- |
| ExpressExpense SRD | [Publisher](https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/) offers 200 receipt images, a roughly 19.2 MB ZIP, and states MIT terms with attribution. It describes images as scanned; complete OCR transcriptions are not supplied on this page. | Candidate for independent receipt-source coverage after checking archive contents, capture type, template duplicates and transcription. Do not label it a verified receipt-photo corpus merely from the download name. Preserve attribution and the downloaded license text. |
| Humans in the Loop receipt labels | [Publisher](https://humansintheloop.org/resources/datasets/free-receipt-ocr-dataset/) describes 192 SRD images, 3,839 manually segmented boxes in ten semantic classes, and CC0 dedication. Access uses a form. | Same underlying SRD parents, not a second independent receipt corpus. Semantic box classes do not establish complete text transcription. Do not send contact information or submit the form as part of this task. |
| WildReceipt | [MMOCR's dataset catalog](https://mmocr.readthedocs.io/en/latest/user_guides/data_prepare/datasetzoo.html) lists English receipt word annotations but dataset license N/A. | Defer training/redistribution pending explicit dataset terms; the code repository license does not establish image rights. No acquisition needed for this phase. |
| COCO-Text v2 | [Author release](https://bgshih.github.io/cocotext/) supplies word polygons, legibility and language attributes. Annotations are CC BY 4.0; underlying images are MSCOCO 2014 train images. | Candidate for new scene/sign parents. Preserve image-specific source rights separately. Select using annotations and human scene review, never OCR confidence or candidate errors. Missing annotations do not establish a text-free image. |

These statements record publisher metadata, not a legal clearance determination. The project has not acquired these new sources or approved a training corpus. All proposed public sources may overlap upstream pretrained model data; source independence from our existing evaluation is a narrower claim than unseen pretraining data.

## Sampling contract

Target a first collection of **160 independent parent captures**: 40 genuine receipt photos, 40 physical-sign scenes, 40 real screenshots with complete transcription, and 40 real text-free images. Use 20 dev and 20 reserved parents per category only if group separation permits; otherwise collect more parents rather than splitting a related group. These are feasibility targets, not a statistical power or production-readiness claim.

Receipt photos should span at least ten merchant/template groups, with oblique views, shadows, folds and low contrast recorded as tags. Scans may be a separate stratum but cannot replace photos. Sign scenes should include oblique/curved text, small text and low contrast, with the difficulty tag recorded by reviewers before inference. Screens should span app/template groups, including light/dark and small type. Real negatives should include paper/tabletops, vegetation, textures and UI panels verified to contain no visible text; unreadable text is not a negative.

Record natural orientation independently of synthetic transforms. Include at least ten naturally rotated text captures in each of the receipt/sign categories, without selecting by model failure. Derived rotations remain attached to their parent and never increase the independent-parent count.

Canonical identity fields are `source`, `source_revision`, `source_id`, original URL and file SHA-256, decoded RGB SHA-256, capture/session ID, merchant/app/location/template group, parent ID, rights evidence, language, capture type, difficulty tags and review status. Avoid personal identifiers in metadata. Receipt images remain local; transcription should flag personal information for collection review rather than silently altering reference text.

## Annotation and acceptance

1. A first human annotator transcribes all visible English text and punctuation, records polygons and reading order, and marks uncertain/illegible regions. A second independent human checks the image without model predictions and adjudicates disagreements. Assistant visual inspection may help triage but does not count as independent human adjudication.
2. Record annotator/reviewer identifiers, dates, disputed regions and resolution. Keep immutable raw labels and versioned corrections. Partial labels must use explicit partial-recall scoring; they cannot enter the complete-transcription stratum. Mixed-language content receives its own tag and is not silently relabeled English.
3. Check exact source IDs and decoded pixels against the pilot, expanded dev, consumed final and every legacy candidate parent. Review perceptual near-duplicates across all sources, including crops, resizes, screenshots and different shots of one document. Group shared templates, locations, sessions and all transformations before splitting.
4. Reserve groups with deterministic SHA-256 ranking using seed `tinyocr-independent-v2`, stratifying by category. Freeze a manifest containing final file/label hashes and the exclusion audit. Fail collection acceptance if required fields, rights records, independent review or category/group coverage are incomplete; never call a provisional split final.
5. Keep reserved images, labels and predictions outside ordinary development commands. Record any model inference on them as consumption. Freeze model, preprocessing, runtime and acceptance protocol before the next final event. Do not repeatedly open the holdout to choose the next candidate.

For full transcripts, retain normalized and case-sensitive CER/WER, exact matches, numeric/currency errors and explicit illegible-region treatment fixed before scoring. Report per category and difficulty tag; do not pool synthetic controls with real captures. For negatives, report images with any text, character counts and raw outputs. Report latency on target devices separately, including initialization and peak memory for deployment work.

## Concrete next acquisition

Download the publisher-linked SRD ZIP with a 32 MiB streaming cap into `data/independent-intake/`, hash it, inspect its license and member inventory without executing contents, and create a local candidate catalog. Check whether the images are photos or scans and identify merchant/template groups using only intake/dev material. If genuine photos or independent human transcripts are unavailable, retain this as an unlabeled candidate source and obtain new captures/review before freezing the 40-photo category. Then inspect COCO-Text metadata and image-rights records for the sign cohort. No model training or new final inference is authorized by this collection design alone.
