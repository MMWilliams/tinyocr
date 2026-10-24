# Pinned HierText evaluator

Unmodified evaluator modules, README and LICENSE from Google's HierText repository,
Git commit `70b6620b2b112597d8219e11eee9773a1403827c`:
https://github.com/google-research-datasets/hiertext/tree/70b6620b2b112597d8219e11eee9773a1403827c

Upstream repository/dataset license: CC BY-SA 4.0; see LICENSE. Attribution:
Shangbang Long, Siyang Qin, Dmitry Panteleev, Alessandro Bissacco, Yasuhisa Fujii,
and Michalis Raptis, *Towards End-to-End Unified Scene Text Detection and Layout
Analysis*, CVPR 2022, and *ICDAR 2023 Competition on Hierarchical Text Detection
and Recognition*, 2023.

TinyOCR calls the polygon evaluator directly, using the existing NumPy/Shapely
runtime and a pinned absl-py overlay. The upstream Apache Beam CLI and notebook
dependencies are not needed. No scoring implementation has been edited. Download
checkpoints and `results/benchmark-sources.json` record source URLs and hashes.

Only scores on HierText use its native dataset protocol. Applying this evaluator
to SROIE regions or TextOCR words is explicitly identified as a shared protocol,
not a reproduction of those datasets' native challenge leaderboards.
