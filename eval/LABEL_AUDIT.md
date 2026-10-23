# Label audit, 2026-10-23

Assistant visual inspection of all 20 legacy scene crops and the 24 development full images. This is a diagnostic review, not independent human reannotation. No upstream transcript was changed, no item was dropped based on model performance, and final screenshots/scenes were not visually reviewed or evaluated.

Legacy crop contact sheet: `runs/scene-review.png`. Candidate parent IDs and annotation IDs are recorded in `legacy_scene_lineage.json`, using exact text lookup in pinned original TextOCR validation annotations. A unique text match is still only a candidate until pixel geometry is checked. All candidate parent IDs, including ambiguous matches, are excluded from the expansion. The full TextOCR validation split is reserved from future training.

| Legacy sample | Visual finding | Action |
| --- | --- | --- |
| scene-0 | MORGAN, unusual source capitalization | Keep; normalized metric handles case |
| scene-733 | EL with a small symbol; source uses registered-mark character | Human review of symbol needed |
| scene-1466 | Printed `3/$5`; source includes an extra backslash | Flag likely annotation escaping error; leave original unchanged |
| scene-2200 | Blurred OUR | Keep difficult crop |
| scene-2933 | EVANS with trailing dot; source bullet punctuation | Flag punctuation convention |
| scene-3667 | branded | Keep |
| scene-4400 | Stylized partial jersey marking; source SYA is uncertain | Flag ambiguous label/crop |
| scene-5134 | SAMSUNG, low resolution | Keep |
| scene-5867 | POWER! with an overlaid marking | Keep difficult crop |
| scene-6601 | STRONG, stylized low contrast | Keep |
| scene-7334 | Calibro with neighboring text included | Flag crop contamination |
| scene-8067 | PHOTOGRAPHY upside down, with another large word in crop | Flag rotation and crop contamination |
| scene-8801 | DRY with decorative quotes and neighboring SUPER text | Flag quote convention and contamination |
| scene-9534 | MOA is partially visible; source crop truncated | Flag incomplete crop |
| scene-10268 | Rosehill | Keep |
| scene-11001 | HIGAN fragment of a longer word | Flag truncation |
| scene-11735 | SNEKUTI with another line above | Flag crop contamination |
| scene-12468 | Complete with dot leaders and another line above | Flag punctuation count and contamination |
| scene-13202 | Curved PILIPINAS on a coin, hard to resolve at crop scale | Keep difficult crop; human confirmation pending |
| scene-13935 | COOL | Keep |

Development screenshot contact sheet: `runs/real-screen-dev-review.png`. These are actual RICO app screenshots, including settings, a chart, sports content, shopping and system dialogs. Status bars, text embedded in photos, and some other visible text are outside the screen annotations. Accordingly, full screens use only labeled-location text-recovery recall, not page CER or precision. Labels were checked byte-for-byte against Google's pinned release before OCR.

Screenshot region inspection exposed two multiline regions (a paragraph in screen-13175 and BUY NOW in screen-2877) that cannot be fairly evaluated by treating every UI crop as one text line. The corrected protocol routes **every** UI-region crop through detection plus recognition, including final regions when eventually evaluated. The original recognizer-only runs remain as `expanded-*-recognizer-diagnostic.json`. Their high error is not a valid single-line OCR benchmark. The corrected manifest changes only region task/domain descriptors and adds a protocol note; all pixels, labels, IDs and dev/final assignments remain identical. One development screen contains Korean text; it remains in the fixed set and is an explicit limitation for the English-first baseline.

Full-scene development contact sheet: `runs/full-scene-dev-review.png`. Covers signs, posters, books, a watch face, a beaker, packaging, clothing and a boxing ring. Small curved/vertical text, occlusion, tiny background logos, multi-column order and upstream transcription uncertainty remain. CER is therefore diagnostic and includes layout-order disagreement; it is accompanied by location-aware exact annotation recovery. Annotation eligibility excludes explicitly illegible-dot labels, which biases this small subset toward legible scenes.

Remaining: human adjudication of flagged labels; pixel-based recovery of legacy crop parent IDs; app/template-level split checks; an independent receipt-photo test set and a larger final benchmark. No training-data clearance is inferred from mirror metadata.
