# ExpressExpense SRD candidate intake

Completed 2026-10-23. **Verified for bounded intake only.** The receipt images remain unlabeled candidates. No independent collection, benchmark split, training approval or new OCR/model result is claimed.

## Acquisition and rights evidence

Downloaded the ZIP linked by the [ExpressExpense publisher page](https://expressexpense.com/blog/free-receipt-images-ocr-machine-learning-dataset/), preserving the page locally. The publisher describes scanned receipts, claims MIT terms and requests attribution to **ExpressExpense.com**. The archive contains **200 JPEG files only**, with no license file, copyright notice, README, labels or transcription. The saved page is evidence of the publisher's claim; it does not establish individual image ownership or capture provenance. These remain collection-review items.

| Evidence | Observed value |
| --- | --- |
| Archive URL | `https://expressexpense.com/large-receipt-image-dataset-SRD.zip` |
| Archive bytes / cap | 19,230,342 / 33,554,432 (32 MiB) |
| Expanded member bytes / cap | 19,263,584 / 268,435,456 (256 MiB) |
| Archive SHA-256 | `aac2e8ea4d42c62b38c2762fc655eafe1e37edb7b67923cce1a494e66bc64939` |
| Actual and publisher MD5 | `c8eb0f2d286da5ab742e7a5b59f15147` — valid 32-character value, exact match |
| Source-page SHA-256 | `aa1a5aafbbf4025af46ee14f8f93435c21bf95acafd5f652f1b89b91bd4676b4` |
| Unique file / decoded-pixel hashes | 198 / 198 |

The previous handoff's uncertainty about MD5 length is resolved by the saved HTML: the publisher's checksum is complete and matches. SHA-256 remains the identity used by the catalog. All images and original evidence stay under ignored `data/independent-intake/srd/`; there was no form submission, contact message, paid request, training or OCR inference.

## Capture and grouping review

Assistant visual triage covered all 200 entries in ten contact sheets. Original-file views additionally inspected `srd-013`, `036`, `054`, `064`, `094`, `157`, `176`, `182`, and `187`. The review source, sheet hashes, per-image notes and limitations are in [eval/srd-review.json](eval/srd-review.json); the generated [candidate catalog](eval/srd-candidates.json) pins that review's hash.

| Provisional capture appearance | Entries |
| --- | ---: |
| Photo-like, capture provenance unconfirmed | 183 |
| Tight crop / mechanism unknown | 14 |
| Scan-like, possibly a processed photograph | 2 |
| Screenshot of a receipt photo, with UI overlay | 1 |

Many images visibly include tables, hands, folds, shadows or perspective. This supports a photo-like triage label, not verified genuine-photo provenance. `srd-013` and `176` have isolated, high-key backgrounds and remain scan-like/unconfirmed. `srd-036` includes social-media UI and personal overlay information; its metadata flags review without copying personal identifiers. Receipt completeness and personal-information review are pending throughout.

The archive also contradicts the publisher's stated minimum image size: **57 entries have a longest side of at most 600 pixels**. Several receipts are cropped, low contrast or partially obscured. French, Italian, Spanish and mixed English/Chinese examples are flagged; English-dominant labels elsewhere are provisional, not full language adjudication.

Stored RGB hashes deliberately do not apply EXIF rotation. Five images have non-identity EXIF orientation: `srd-040` and `182` use 6, `094` uses 3, and `096` and `157` use 8. Contact sheets show stored pixels; the original-file viewer can honor EXIF. Their apparent right-angle rotation is therefore **not evidence of naturally rotated capture coverage**. Source files remain byte-identical.

| Group | Evidence and handling |
| --- | --- |
| La Source: `035`, `042`, `115` | Identical files and decoded pixels. One provisional parent, `srd-035`. |
| Primo: `054`, `064`, `187` | Same printed receipt, folds and wood grain across crops/resizes, reviewed at original resolution. One provisional parent, `srd-054`. |
| Repeated merchants | In-N-Out (`019`, `065`), Moonstar (`023`, `134`), McDonalds (`026`, `101`), Wings and Things (`033`, `056`), Pappadeaux (`037`, `167`), Panda Express (`038`, `120`), Miguels Mexican (`146`, `190`), plus the two groups above, share conservative merchant clusters. |

There are **196 provisional parent groups**, not 196 proven independent captures. The review records 172 named merchant clusters and leaves 17 entries with unresolved merchant identity. Merchant clusters are only a first grouping pass: shared point-of-sale templates across merchants, source sessions, more near-duplicates and cross-corpus overlap remain unreviewed. No grouping is approved for a split. The 40-photo category and naturally rotated quota remain unfulfilled.

## Reproduction and verification

From the project root, using the locked Python 3.12.9 environment:

```powershell
uv run --locked python -m tinyocr.intake --verify-only
uv run --locked python -m unittest discover -s tests -v
uv run --locked python -m tinyocr.evaluate_robustness --verify-only
uv run --locked python -m tinyocr.evaluate_final --verify-only
git diff --check
```

`python -m tinyocr.intake` performs first acquisition when no report exists; once complete, it verifies offline. `--contact-sheets` recreates local review sheets. After a deliberate edit to the manual review file, `--apply-review` rebuilds the catalog offline while requiring unchanged acquisition evidence and archive inventory. Missing or corrupt accepted evidence is an error, not a silent redownload. Complete page/archive transfers are checkpoints; interrupted partial bodies are discarded and restart from byte zero. A crash between file promotion and checkpoint creation leaves an explicit uncheckpointed-file error requiring inspection.

The downloader reads and writes at most the configured body cap, rejects encoded/ambiguous transfers and declared/truncated-size mismatches, and rejects unknown-length streams that hit the cap without proven EOF. ZIP checks bound member count, member bytes, total expansion and image pixels; reject unsafe names, aliases, links and encryption; and validate member CRCs. Only decoded image candidates are copied to filenames derived from hashes. Archive paths are never extracted or executed.

The [verification matrix](INTAKE_VERIFICATION.md) records the acceptance checks and outcomes. Offline reconstruction verifies the source and archive checkpoints, every member, every saved image, stored RGB hashes, review provenance and candidate-only status. Repeating preparation and verification with network session creation disabled preserved both catalog and report hashes. Complete transcripts, independent human adjudication, confirmed capture types, rights review and cross-corpus deduplication are collection acceptance gates outside this completed intake.

Final checks passed: **53 tests, zero skipped**, including 12 intake tests; the intake verifier; the unchanged 80-item-per-model robustness evidence; the unchanged 48-item-per-model final evidence; and the whitespace check. No model inference ran during intake or these evidence-verification commands.

Next source work: inspect COCO-Text v2 annotation metadata and image-specific rights, with a bounded acquisition/verification plan before downloading sign images. In parallel with future collection, the SRD candidates need human review and full transcription before any partition is frozen. Existing model locks and consumed final results remain unchanged.
