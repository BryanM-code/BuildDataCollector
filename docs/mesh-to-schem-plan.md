# Mesh → .schem: dataset, voxeliser, model and evaluation plan

**Current scope (2026-10-11): schematic-first pivot.** Human-built schematics are now the
training targets and model inputs are derived from them. Training is in scope once the
dataset is frozen and paired (Phases 2–4). [Version 4](#version-4-2026-10-11-schematic-first-pivot)
is the current plan; sections 0–11 below are the version 3 record and are marked where
superseded.

---

## Version 4 (2026-10-11): schematic-first pivot

Version 3 used Objaverse meshes as the source, made targets with `mcvoxel` rules, and kept
real builds for evaluation only. Version 4 reverses that: **human builds are the targets**
and each input is generated from its target.

### Pipeline

```
inference   photo → background removal → TRELLIS mesh → normalize → voxelize at 2x block resolution → learned model → .schem
pairs       human .schem (target) → coloured surface mesh → synth/degrade.py (TRELLIS-like) → voxelize at 2x into the target's frame → input
```

### Dataset

| | |
| --- | --- |
| Builds | 321 human-built realistic schematics (Planet Minecraft 235 + Abfielder 86), frozen 2026-10-11 as **real-schems-v1** |
| Location | `data/real-schems-v1/` (local only): read-only `source/` copy plus `index.json`, `splits.json`, `near_duplicates.json`. See [real-schems-v1.md](real-schems-v1.md) |
| Collected at | `C:\Users\mendo\Documents\Codex\2026-10-05\find-and-download-30-high-quality-2\outputs\Real2Minecraft_Dataset\` (never written to) |

### Decisions (2026-10-11; D15 revised and D20–D24 added 2026-10-12)

- **D13 Data direction: schematic-first.** Human builds are targets, inputs are derived from
  them. The Objaverse synth path is parked (kept, not deleted, possible pretraining source
  later).
- **D14 Licensing (supersedes D6 for the schematic dataset): personal research prototype.**
  - Train privately on tiers A+B+C. Never redistribute schematics, derived pairs or renders.
  - Tier A = explicit reuse permission. Tier B = licence unclear. Tier C = personal-use /
    noncommercial. Tier X = excluded.
  - Tier X now: Costanera (CC BY-NC-ND, no derivatives) and the Xorph barn (no
    redistribution).
  - Any public release or commercial use of model weights requires retraining on tier A
    only, or re-auditing licences first.
  - Every build keeps its provenance and licence fields.

  Details: [dataset-licensing.md](dataset-licensing.md#schematic-dataset-2026-10).
- **D15 Scale (revised 2026-10-12; supersedes D5 for this dataset): no scale rule.**
  - Never rescale or shrink a human-built schematic.
  - Never filter by storey height. Buildings legitimately vary in scale, sometimes on purpose.
  - The storey estimate (blocks per storey) stays in the normalized records as information
    only.
  - The first version of D15 set aside builds outside a storey-height band. That is
    withdrawn.
  - Storey-height conditioning is still a possible later option.
- **D16 Input resolution:** input voxels at 2x the target grid (each block = 2x2x2 input
  cells), same frame and origin as the target.
- **D17 Evaluation (supersedes D9):** split by creator ~80/10/10 on the new dataset, plus a
  small real-photo TRELLIS set. The old `eval/real-eval-v1.json` (64 builds from the earlier
  122-build library) is marked superseded and gets reconciled during the dataset freeze.
- **D18 Baseline:** the current rule-based mcvoxel/building output is the baseline the model
  must beat. It stays in the app until the model beats it.
- **D19 (2026-10-11): The old BDC library (`data/builds/`, 122 builds) is ARCHIVED, not
  merged.**
  - 96 of its 106 real builds lost their block states on import (stairs/slabs in default
    orientation).
  - Only 1 of 106 has a source URL, so licence tiers can't be assigned.
  - The 16 synthetic houses stay excluded.
  - `eval/real-eval-v1.json` stays on disk (tests use it) but is superseded by the D17
    creator split.
  - Do not touch `data/builds/`.
- **D20 (2026-10-12): Vocabulary v3 (`mcvox-v3`).** Shapes and materials expand so human
  builds are expressed faithfully, including every texture variant (wood species, glass
  colours, stone variants).
  - Factored shape × material. A material is a texture family, and a `VALID[shape,
    material]` table records which pairs exist: 30 shapes × 301 materials, 1,210 valid
    pairs.
  - Append-only: v2's shape and material IDs are unchanged and v2 targets still decode.
  - New shapes: upside-down stairs, pane, closed and open trapdoors, the lower door half,
    horizontal pillars, fence gates and carpet.
  - The model predicts only what the game cannot derive. Stair corners, connections, door
    upper halves and hinges, and gate `in_wall` are rebuilt after assembly.
  - Remaining unrepresentable blocks in supervised regions become IGNORE in the training
    targets, never air. That covers decor, fluids, snow layers, unsupported blocks and door
    upper halves.
  - Waxed, infested and petrified blocks are read-time aliases, not vocabulary.
  - Spec: Real2Minecraft `mcvoxel/README.md` ("Vocabulary v3"). Coverage:
    [vocab-coverage.md](vocab-coverage.md).
- **D21 (2026-10-12): Target Minecraft version for validation and export is 1.21.11
  (DataVersion 4671).** It is the newest 1.21.x release.
  - Pale oak needs 1.21.4 or later.
  - It covers every block real-schems-v1 uses except `poplar_fence` and
    `stripped_poplar_wood`. Those come from one build (`pmc-house-32`, DataVersion 5023,
    Minecraft 26.3), and no 1.21.x has them, so they are IGNORE.
  - `chain` is read as `iron_chain`, its name since 1.21.9.
  - Block table: Real2Minecraft `mcvoxel/blocktables/blocks_1_21_11.json`, from PrismarineJS
    minecraft-data (MIT; source URL, commit and SHA256 kept in the file).
  - `schem_write` validates against it whenever it writes DataVersion 4671.
- **D22 (2026-10-12): Exterior includes every open part of the building.** These are part
  of the building and trained on:
  - porches and recessed porches, verandas, balconies (recessed ones too), loggias;
  - arcades, colonnades, carports, canopies;
  - eaves and soffits, gable undersides, jettied overhangs;
  - window and door reveals.

  Interior means enclosed rooms only. Being covered from above never, by itself, makes air
  interior.
  - Implemented as an opening-size rule (`schemset/interior.py`). The outside reaches through
    openings at least 3 wide and 2 tall, plus 1 block through smaller ones, plus open gaps.
  - Rooms behind doorways or windows up to 2 wide stay interior, glass or no glass.
  - Big openings into halls are accepted as exterior and flagged `large_opening`.
  - **Amended 2026-10-14: depth under cover.** Covered exterior air is supervised only near
    the outside.
    - Exterior air more than `COVER_DEPTH` = 6 steps from uncovered exterior air becomes
      interior. Steps are 6-connected, through exterior air; uncovered air has nothing above
      it, and the grid boundary counts as uncovered.
    - The shell is taken afterwards.
    - The audit uses the same depth.
    - Being covered never on its own makes air interior; only depth does.
    - Builds where the cap turned more than 200 air voxels interior are flagged
      `deep_cover_capped`.
    - Why 6: [real-schems-v1.md](real-schems-v1.md#depth-cap-under-cover-d22-amended-2026-10-14).
- **D23 (2026-10-12): Ground comes only from natural terrain.**
  - Building materials are never landscape because of their name.
  - When unsure, keep the block: a kept stray block is cheaper than a hole in a building.
  - Natural terrain is a layered ground mass connected to the bottom and edges of the
    capture, with natural evidence.
  - Its surface gives each column a ground height. With no natural terrain around the
    building there is no ground.
  - Only blocks genuinely buried below that surface (a foundation course, a basement) are
    masked as below ground.
  - If more than 15% of the kept blocks would be, the ground is dropped (`ground_fallback`).
  - Code: `schemset/crop.py`.
- **D24 (2026-10-12): The two landscape dioramas are set aside:** split "excluded", flag
  `excluded_landscape_diorama`.
  - `pmc-traditional-house-cottage-style`: a forest and a road kept with the house.
  - `pmc-the-lighthouse-5778696`: a rock island.
  - Applied in `schemset.freeze.derive` (`SET_ASIDE`) after the seeded split, so no other
    build moves.

### Phase plan

| Phase | Work |
| --- | --- |
| 1 (2026-10-11) | Record the pivot in the docs; park the old pipeline (documentation only) |
| 2 (done 2026-10-11) | Freeze and dedupe the dataset; reconcile `eval/real-eval-v1.json` (D17). **Done:** 321 builds frozen as [real-schems-v1](real-schems-v1.md) (1,792 files verified); tiers A 43 / B 265 / C 11 / X 2; creator split train 255 / val 32 / test 32 / excluded 2 (216 creator groups, seed 20261011); 0 metadata near-duplicate candidates; `real-eval-v1` superseded (D19) |
| 3 (done 2026-10-12, measured; nothing decided) | Normalize: `.litematic` reader, legacy block IDs, crop, vocabulary coverage. **Done:** all 319 non-excluded builds read and validated: 0 failures; dims and the 178 metadata non-air counts all match; 0 unknown legacy ids. Cropped to the building: 48% of non-air blocks were landscape or site; 85 `crop_aggressive`, 18 `multi_building`, 3 `crop_kept_landscape`, 2 `crop_no_core`. Interior masked as ignore: median interior fraction 0.86; 17 `interior_leak_suspect`. Storey height recorded (information only, D15): 142 builds at high/medium confidence, 119 low, 58 unknown. **Vocabulary coverage of supervised blocks (v2): 72.7% exact, 3.2% lossy, 12.2% missing material, 11.2% missing shape, 0.7% decor.** Top 20 materials plus pane, door, trapdoor and upside-down-stair classes would reach 94.7% ([vocab-coverage.md](vocab-coverage.md)). Details: [real-schems-v1.md, Normalization](real-schems-v1.md#normalization-phase-3). The geometric duplicate check (rotation, crop, cross-format) deferred from Phase 2 is **not done yet**: it is now possible, as every build parses |
| Vocab v3 (done 2026-10-12) | Vocabulary v3 (D20), target version (D21), no scale rule (D15 revised). **Done:** `mcvox-v3` has 30 shapes × 301 materials and 1,210 valid pairs; v2 IDs are unchanged. Every `VALID` decode validates against 1.21.11. Every normalized build now carries training targets (`target_shape` / `target_material` in its npz; IGNORE for interior, decor, unsupported blocks and door upper halves). **Coverage of supervised blocks, v2 → v3: 75.9% → 98.5% representable** (96.15% exact, 2.17% lossy, 0.18% derived, 1.49% IGNORE: 1.48% decor / fluids / snow, 0.015% structural). By split: train 76.6% → 98.4%, val 74.9% → 98.6%, test 69.7% → 99.2% ([vocab-coverage.md](vocab-coverage.md)) |
| 3 fixes (done 2026-10-12) | Interior/exterior rule (D22), ground detection and crop correctness (D23), dioramas set aside (D24), duplicate check, re-run. **Done:** porches, overhangs and balconies are trained on. Outside-visible but ignored is now median 0.2% (p90 3.2%), against 11.5% (p90 25.5%) before; builds over 25% fell from 34 to 0. Ground per column from natural terrain only: the six ground-bug builds are at 0% below ground (Kaiserthal was 95%). Builds over 15% below ground fell from 15 to 0. The cemetery wall, the gym hall's missing floor and the bakery's frame logs are fixed. Supervised shell blocks are up 52% (+32% in builds without `large_opening`). Duplicate check after the crop: 0 duplicates in 317 builds, no split moves. Splits: train 253 / val 32 / test 32 / excluded 4. v3 coverage 98.44% representable. Flags for review: 69 `large_opening`, 4 `ground_fallback`, 10 `multi_building`, 3 `site_linear_removed`, 8 `crop_kept_landscape`. Details: [real-schems-v1.md](real-schems-v1.md#phase-3-fixes-before-and-after-2026-10-12), [phase3-fixes.md](phase3-fixes.md) |
| D22 cap (done 2026-10-14) | Depth cap under cover, `COVER_DEPTH` = 6 (normalize-4). **Done:** open halls, hollow facade shells and open ground floors no longer train their insides as facade. The hall set's interior fraction is now 0.79–0.89 (was 0.31–0.48); the brick tenement stays at 0.30, as its hollow is open sky. The porch set's visible-but-ignored stays within +0.2 points. Supervised shell blocks 3.27 M → 2.73 M. `large_opening` 69 → 11; `deep_cover_capped` 108. v3 coverage 98.51% |
| 4 (done 2026-10-14) | Generate pairs (D16); real-TRELLIS check of the synthetic inputs. **Done:** 317 builds → **1,268 pairs** (4 variants each; train 1,012 / val 128 / test 128), **0 rejected**, 0 builds failed; 283 MB, 23.5 min on 6 workers. Input = the build's outside only (`schemset.meshify`: real 1/16 block geometry, no interior, no ground plate) → ±5° yaw / ±0.5 block offset → `synth.degrade` → voxelised at 2× in exactly the target frame. Block-level IoU with the target volume p5 / p50 / p95 0.54 / 0.77 / 0.88; frame residual against the clean placed reference ≤ 1.16 cells; 26 pairs in 9 builds kept and flagged `low_iou` (blobs bigger than open or sprawling builds). Domain-gap guard: unsmoothed inputs were blockier than real TRELLIS, so the bottleneck blur got a 1-block floor (σ 0.42); size-matched, stair45 0.21 vs 0.25 and roughness 7.8° vs 8.1°. Colour does not match: synthetic L* 46 vs 22. Real-TRELLIS set: 25 val/test builds chosen and rendered; TRELLIS server down. Format and loader contract: [pairs-v1.md](pairs-v1.md) |
| 4 follow-up (done 2026-10-15) | Colour recalibration, real-TRELLIS eval sets, eval alignment. **Done:** **pairs-2**, 1,268 pairs regenerated (27.6 min, 279 MB, geometry identical to pairs-1: seeds frozen). The colour model is fitted to TRELLIS on real photos: log-uniform exposure, a directional sun, muted chroma, dark cool glass. L\* p10 / p50 / p90 went from 34 / 46 / 63 to **12 / 22 / 37** (reference 11 / 22 / 34, 78% inside its band); C\* p50 from 10.8 to **6.0** (reference 6.7, 79% inside). Real-TRELLIS eval on 25 val/test builds, two sets: **smooth photos** (the main set; σ 0.6 chosen by trials) give TRELLIS stair45 0.22 and roughness 7.7°, like TRELLIS on real photos (0.25 / 8.1°); **blocky photos** (the stress test) give 0.88 / 18.4°. Eval alignment (±16-cell shift search, centred where the validator measures, voxelised once and cropped): misaligned 16 → **0**, residual p90 4.0 → 0.47 cells; 1 build fails in both sets because TRELLIS misses most of it. [pairs-v1.md](pairs-v1.md) |
| 5 | torch environment, 3D U-Net, baseline score (D18) |

Open questions (for you; nothing below is decided).

Resolved 2026-10-12:

* the vocabulary (D20)
* storey-height filtering: none (D15 revised)
* `poplar_fence` / `stripped_poplar_wood`: a Minecraft 26.3 wood, newer than any 1.21.x, so
  IGNORE under D21
* the interior heuristic: exterior by opening size (D22)
* below-ground blocks: only genuinely buried blocks, ground from natural terrain only (D23)
* the two landscape dioramas: set aside (D24)
* geometric duplicates: checked after the crop, none found

Still open:

1. **Compounds and slope houses.** Every building of the 10 `multi_building` builds is kept in
   one target. Should they be one target each, split per building, or set aside? The 4
   `ground_fallback` builds are built into a slope, and the Richelieu cliff house loses the
   rock that formed its rooms' back walls. Keep, re-crop by hand, or set aside?
2. **Depth cap.** `COVER_DEPTH` = 6 ignores covered exterior air deeper than 6 steps. That
   includes the middle of big canopies and long covered passages, not only hall interiors.
   Review the 108 `deep_cover_capped` builds in the gallery.
3. **Invisible blocks.** `light`, `barrier` and `structure_void` became air. Is that right?
4. **D21's wording.** No 1.21.x covers every block in the dataset: poplar (87 blocks, one
   build) is 26.3. Should 1.21.11 with poplar as IGNORE stay the target, or should it move to
   26.3 (DataVersion 5023)?
5. **Vocabulary append.** The crop now keeps ore, gravel and blackstone used as building
   material, so 12 more structural blocks meet D20's data rule (2+ builds or 200+ blocks):
   - gold, emerald and redstone ore; emerald block
   - chiseled polished blackstone
   - suspicious gravel, rooted dirt, red sand
   - shroomlight, honey block, bee nest, bamboo mosaic

   Together they are 2,602 supervised blocks (0.08%), now IGNORE. Append them to v3?
   ([vocab-coverage.md](vocab-coverage.md#structural-blocks-still-outside-the-vocabulary))
6. **Eval-photo colour.** Input colour itself is resolved in pairs-2 (synthetic L\* p50 22, the
   same as TRELLIS on real photos). The real-TRELLIS eval photos, though, are renders in flat
   thumbnail colours, and TRELLIS bakes them bright (L\* 39–44). Options:
   - score materials on the eval sets knowing that;
   - normalise lightness per input, the same at inference;
   - render the eval photos with realistic exposure.

   ([pairs-v1.md](pairs-v1.md#trellis-statistics-per-set-p10--p50--p90))
7. **`low_iou` pairs.** 26 pairs in 9 builds are kept and flagged. The input blob is far
   bigger than the build: open barns, a sprawling colonial complex, the 9-storey Yaroslavl
   block. Keep them, down-weight them, or hold them out of training?
8. **Real-photo replica set.** Ten candidate builds have free-licence Wikimedia Commons photos
   identified; three of them are in test or val. Approve the downloads
   ([pairs-v1.md](pairs-v1.md#real-photo-replica-set-optional-pending))?

Which code is parked, active or excluded:
[dataset-pipeline.md, Status](dataset-pipeline.md#status-2026-10-11).

---

*Version 3, 2026-09-26. Version 2 (2026-09-24) proposed the plan and ran pilot v1; version 3
records your V1 decisions (section 9), the dataset pipeline built on them and pilot v2
(section 6.6). The pipeline itself is documented in [dataset-pipeline.md](dataset-pipeline.md).
No model code has been written.*

Companion documents:

- [dataset-pipeline.md](dataset-pipeline.md) — the V1 preprocessing pipeline: stages, reason
  codes, configuration, outputs, commands, known limitations.
- [dataset-licensing.md](dataset-licensing.md) — the licence register, written before any
  mesh was downloaded.
- `eval/real-eval-v1.json` — the held-out real evaluation set (64 builds, one per family).
  Superseded by D17; kept on disk for its tests (D19).
- `data/synthetic/pilot-v2/` (gitignored) — pilot v2: pairs, report.md, contact sheets,
  review tool data. `pilot-v1/` is the earlier run.

Sections 0–8 describe version 2's findings and pilot v1; where V1 decisions changed them
(1:1 scale, no photogrammetry, builder roofs, the fixed voxeliser), section 6.6 and section 9
say so.

---

## 0. Where things stand

*Superseded by [version 4](#version-4-2026-10-11-schematic-first-pivot): this is the version 3 status.*

| Area | Status |
| --- | --- |
| Licensing | Register written first. Policy: **CC0 + CC BY 4.0 only**, all five uses open with attribution. |
| Source dataset | Objaverse 1.0 indexed from metadata: **21,584** building objects pass the filters, **14,804** of them permissively licensed. Thin in exactly the categories you care most about. |
| Shared code | `mcvoxel` package extracted inside Real2Minecraft and installed into this repo. Real2Minecraft's outputs verified **byte-identical**. |
| Real library | **0** usable (mesh, schem) pairs today. 106 real builds from **64** independent sources, 78% detached houses. Recommended: **evaluation only**. |
| Large buildings | Tiled inference designed and tested; 64³, 96³ and 128³ windows all fit the 4060 Ti. |
| Pilot | **200** diverse objects → **166** pairs after automatic QA, in about 8 minutes. Hand audit: **57% clean, 22% flawed, 21% junk**. |
| Degradation | Modelled on TRELLIS's own bottleneck and calibrated against 54 real TRELLIS meshes. |

The pipeline works end to end at a cost that scales (10 s and 166 KB per pair). Two things
must be settled before a full run: how much junk is acceptable (section 6.4), and whether
targets should follow geometry or builder convention for roofs (section 6.3).

---

## 1. Your decisions, and what each changed

*Row 5 superseded by D13 and D17 ([version 4](#version-4-2026-10-11-schematic-first-pivot)): real builds are now the training targets.*

| # | Decision | What it changed |
| --- | --- | --- |
| 1 | Moderately messy, TRELLIS-like inputs; clean targets | Degradation reproduces TRELLIS's measured imperfections (section 7), calibrated on real TRELLIS output. |
| 2 | Permissive licences; flag restrictions; document before downloading | Licence register per use (training, derived data, code, examples, model). Pilot restricted to CC0 + CC BY. |
| 3 | Any building size; chunked processing; 64³ first, test 96³/128³ | Windowed inference with blended reassembly; GPU benchmark at all three sizes (section 5). |
| 4 | Shared package extracted from Real2Minecraft | `mcvoxel` created with `git mv`; reuse, modification and gaps listed in section 3. |
| 5 | Real builds measure the gap; do not assume a 30/50 split | Library inspected; evaluation-only recommendation from the numbers (section 4). |
| 6 | Diversity is a core requirement | Taxonomy of 14 categories; pilot quotas put suburban houses at 6 of 200. |
| 7 | Mesh → blocks with block type, orientation, slabs, stairs | Target vocabulary is material × shape including stair facing (section 5.1). |
| 8 | Treat the synthetic→real gap as something to measure | Evaluation protocol on real builds (section 4.4); the first measurable gap (stairs) already found (section 6.3). |

---

## 2. Source datasets, re-evaluated for architectural diversity

*Parked by D13 ([version 4](#version-4-2026-10-11-schematic-first-pivot)): Objaverse is no longer the training source; kept as a possible pretraining source.*

### 2.1 Objaverse 1.0, measured from metadata

All 160 metadata shards (798,759 objects) were classified from TRELLIS-500K render captions
(or name and tags when there is no caption) into 14 categories. The rule reads the
caption's **head phrase** — "A two-story brick house with a garage" is a house — and rejects
objects whose head noun is not a building ("A school desk…", "An SUV with a bull bar…").
Rejection funnel over 40,437 candidates:

| Reason | Objects |
| --- | --- |
| **Passes (pool)** | **21,584** |
| Stylised / cartoon / fantasy (text or Objaverse++ style) | 4,960 |
| Furnished interior | 2,310 |
| Low Objaverse++ quality (< 2 of 3) | 2,217 |
| Scene (city, street, diorama) | 2,148 |
| Object, not building | 1,211 |
| Too few faces | 1,044 |
| Several buildings | 791 |
| Interior view | 744 |
| Voxel / Minecraft / LEGO in text | 713 |
| Ruin | 648 |
| Rip of a commercial game asset | 594 |
| Other (animated, toy, unfinished, cutaway, part) | 1,473 |

Objaverse++ marks most buildings-on-a-plot as "scene" or "multi-object". A hand sample
showed they are usually one building with a garden, pool or cars, so those flags **mark**
an object (`pool_kind = plot`) instead of rejecting it; the voxeliser removes the plot.

### 2.2 Coverage and quality by category (permissive licences only)

Precision = share of a random sample of about 20 per category that I judged, from the
caption and thumbnail, to be one real building (rounded; not checked for pavilion and office).

| Category | CC0 + CC BY | With caption | Precision | Scan share | Median faces | Median GLB |
| --- | --- | --- | --- | --- | --- | --- |
| House | 4,649 | 2,750 | ~100% | 16% | 209k | 16.7 MB |
| Generic building | 3,281 | 2,399 | ~100% | 20% | 112k | 13.2 MB |
| Religious | 1,520 | 583 | ~85% | 64% | 205k | 26.3 MB |
| Institutional | 1,047 | 77 | ~75% | 53% | 226k | 24.4 MB |
| Tower | 981 | 641 | ~85% | 35% | 40k | 10.1 MB |
| Fortification | 769 | 350 | ~75% | 59% | 211k | 25.5 MB |
| Industrial | 722 | 307 | ~90% | 33% | 52k | 14.7 MB |
| Mid-rise (generic, 4–14 storeys) | 645 | 644 | ~100% | 12% | 159k | 14.6 MB |
| Commercial | 551 | 115 | ~90% | 33% | 77k | 11.4 MB |
| **Skyscraper** | **241** | 188 | ~100% | 14% | 48k | 4.4 MB |
| **Apartment** | **190** | 102 | ~90% | 13% | 336k | 22.5 MB |
| Pavilion | 144 | 92 | — | 23% | 46k | 8.9 MB |
| **Townhouse** | **34** | 8 | ~100% | 23% | 95k | 15.6 MB |
| **Office** | **30** | 12 | — | 24% | 137k | 9.7 MB |

Findings that matter for diversity:

- **Multi-family, office and high-rise are thin.** Townhouses (34), offices (30),
  apartments (190) and skyscrapers (241) are two orders of magnitude below houses. The
  645 generic mid-rises are mostly apartment and office blocks without the word in the
  caption.
- **One uploader is 25% of the whole pool.** `homedesign3d` contributes 5,458 objects, all
  CC BY-NC-SA and nearly all houses on landscaped plots. The licence policy excludes them;
  so would any diversity cap.
- **Institutional, religious and fortification lean on scans** (53–64%), and in the pilot
  scans were junk 6 times in 11.

### 2.3 Second source for real-scale, multi-family and high-rise geometry

**Berlin LoD2** (verified on tile `LoD2_392_5820`): about 925 CityGML tiles under
Datenlizenz Deutschland – Zero 2.0 (no conditions at all), with a cadastral
**building-use code** per building, roof-type codes and a measured height in metres. No
textures and no windows or doors. It is the natural source for
apartment blocks, offices and towers at real scale as **geometry-only** samples: the shape
loss applies, the material loss is masked. Not implemented yet (a CityGML reader is the
missing piece) — Decision D7.

Others considered: 3D BAG (CC BY 4.0, untextured, no use labels), Helsinki (CC BY 4.0,
textured), PLATEAU (PDL 1.0 plus a Surveying Act caveat to check per dataset). Rejected:
ShapeNet (no buildings), 3D-FRONT (interiors only), Objaverse-XL GitHub (no per-object
licences), BuildingNet (3D Warehouse origin, no licence text), Building3D (NC-SA),
BuildingWorld and ArchiSet (not available).

### 2.4 Recommendation

Objaverse 1.0 (CC0 + CC BY) as the primary textured source, **with category quotas**, not
proportional sampling; Berlin LoD2 as a geometry-only second source for multi-family and
high-rise at real scale.

---

## 3. The shared package: `mcvoxel`

### 3.1 What moved, what changed, what is new

The package lives at `Real2Minecraft/mcvoxel/` (installable via its `pyproject.toml`,
which exposes only `mcvoxel`) and is installed editable here through
`requirements-synth.txt`. The move used `git mv`, so file history follows.

| Module | Origin | Status |
| --- | --- | --- |
| `voxelize` | R2M | **Reused**; loop rasteriser replaced by `raster` (identical output); `_clean_voxels` gained an opt-in `pad_border` |
| `voxel_sampling` | R2M | **Modified**: general `sample_voxel_surface` for any colour source, per-voxel material tallies, occupied-voxel accumulation, chunked batches. The single-texture function is now a thin wrapper with bit-identical output. |
| `block_shapes` | R2M | **Modified**: vectorised `octant_counts` (identical output) |
| `block_palette`, `color_quantize`, `color_repair`, `mesh_normalize`, `schematic` | R2M | **Reused unchanged** |
| `yaw_align` | R2M `building/` | **Reused**; default parameters added |
| `raster` | new | Vectorised surface rasteriser with explicit grid frame; 30–110× faster than the loop |
| `scene` | new | Multi-material GLB loader: per-face material, glass/door kinds |
| `grid` | new | Scene → voxel model at any size: explicit scale and frame, ground plate removal, enclosure of leaky interiors, per-voxel shape classification in chunks, visible shell |
| `assign` | new | Deterministic blocks for a target: R2M's colour chain plus glass and door rules |
| `vocab` | new | Factored vocabulary (8 shapes × 120 materials), encode/decode |
| `blockstates` | new | Rotation/mirror with stair-facing remap; the game's stair-corner rule |
| `tiling` | new | Windowed inference and blended reassembly |

**Verification.** After every change, Real2Minecraft's building-mode golden
(`dev_building.py --golden check`, 45 files) stayed clean and sculpture-mode voxelisation
of six cached TRELLIS meshes in both colour modes (12 cases) stayed **bit-identical**. As a
side effect those 12 voxelisations dropped from 4 min 36 s to 21 s of wall time. `Real2Minecraft/tests/`
holds 18 tests for the package.

### 3.2 Findings about the existing code

- **`_clean_voxels` strips the outermost layer of every grid.** scipy's `binary_closing`
  treats outside the array as empty, so erosion removes solid voxels on the boundary: a
  solid 6³ cube comes back as 4³. Real2Minecraft's grids are the mesh's own bounding box,
  so every sculpture build loses its outer layer on all six sides and those surfaces fall
  back to closest-point colours. Not changed in Real2Minecraft (it would change the app's
  output and golden); the dataset pipeline passes `pad_border=True`. Decision D11.
- **Stair facing is now verified.** Real2Minecraft's notes said facing was never checked
  in a real paste. In the 106 real builds, for 42,233 bottom-half stairs, the block one up
  and one toward `facing` is solid 84.0% of the time, and one up and away 5.0%: `facing`
  names the high side, which is what the classifier emits.
- **Thin-plane models need two new rules.** Architectural models often have zero-thickness
  roofs, open bottoms and glassless windows. The global fine flood leaked under such roofs
  and turned them into slabs, so inside/outside is now decided per voxel from its own
  air-facing sides; and empty voxels covered from above and from four of five other
  directions count as enclosed (rooms yes, courtyards and porches no).
- **Real2Minecraft has no git remote.** The Mac cannot install `mcvoxel` until it has the
  folder. Decision D10.

---

## 4. The real Minecraft library

*Superseded by D13 and D17 ([version 4](#version-4-2026-10-11-schematic-first-pivot)): the evaluation-only split (4.3) is reversed; counts describe the earlier library.*

### 4.1 What exists

| Measure | Value |
| --- | --- |
| Real builds (synthetic grammar houses excluded) | 106 |
| With a TRELLIS mesh or AI-generated image | **0** |
| With at least one screenshot | 106 (102 with one, 4 with two) |
| Independent sources ("families": same author or pack) | **64** (13 multi-member families hold 55 builds) |
| Byte-identical duplicates | 1 pair (`7__grannitlune`, `81__grannitlune-1`) |
| With a recorded source URL | 1, and it is PlanetMinecraft's front page |

By category, from renders of all 106:

| Category | Builds | Independent sources |
| --- | --- | --- |
| Detached house (suburban, modern, mansion) | 83 | 49 |
| Rustic, medieval, cabin, village | 14 | 9 |
| Urban townhouse / row | 7 | 4 |
| Commercial | 2 | 2 |
| Apartment, office, tower, skyscraper, civic, religious, industrial | **0** | **0** |

### 4.2 Convertibility

Every build has the screenshot the AI-photo step starts from, but the screenshots are
source-site images of mixed usefulness (in a random sample of eight: four clear full views, two of them with a
neighbouring build at the edge; three partly hidden by trees or taken from far away; one
web capture with a title across it). The
app's own Inspect render is a usable fallback for any build. 17 builds carry analyser
flags (12 terrain taper, 4 deep terrain, 1 floating gap) and should be cropped before capture.
None has been converted, so conversion success is unmeasured; each conversion costs one
manual AI photo plus a 1–2 minute TRELLIS run.

### 4.3 Recommended split: evaluation only

The collection is better suited to evaluation than to fine-tuning:

1. **No pairs exist**, so any split is future manual work.
2. **It is 78% detached houses** from 64 sources. Fine-tuning on it would pull a model meant
   for all architecture toward suburban conventions — the opposite of Decision 5.
3. **Its licences are unknown.** Keeping it out of training keeps a published model clean
   of it (see the register).
4. **It is the only way to measure the synthetic→real gap**, and only a held-out set does
   that.

Proposed **real-eval-v1**: one build per family, **64 builds** (64 manual conversions). If
that is too much, the smallest useful set is every non-detached family (15) plus 20
detached families stratified by roof type, storeys and size: **35 builds**. No fine-tuning
pool for now; revisit only if the measured gap is large, and then with *new* builds.

**Future data-collection gap:** the library has no real Minecraft build of an apartment
block, office, tower, skyscraper, church, school or factory. Candidate sources, licences
unchecked: BuildTheEarth (1:1 real buildings in Minecraft), PlanetMinecraft builds with an
explicit licence, or your own builds.

### 4.4 How the gap will be measured

Each real build: register the TRELLIS mesh to the schem (yaw 0/90/180/270 ± 5°, XZ ± 3,
scale ± 10%, maximising occupancy IoU); map its blocks into the vocabulary; score only the
visible shell and outside air, ignoring terrain, plants and interior. Metrics: occupancy
IoU, shape accuracy on the shell, material and material-family accuracy, plus the
distribution checks `house_stats` already computes (roof type, storeys, stairs and slab
share). The same metrics on held-out synthetic pairs give the gap directly.

---

## 5. Representation, large buildings and compute

### 5.1 Target encoding

*Scale bullet superseded by D15, input frame resolution by D16 ([version 4](#version-4-2026-10-11-schematic-first-pivot)).*

- **Vocabulary:** 8 shapes (air, full, bottom slab, top slab, stairs facing E/W/N/S) × 120
  materials (the 110-block palette, 9 glasses, air). Upside-down stairs, panes, fences,
  doors and trapdoors are v2.
- **Supervision:** the visible shell (solid voxels touching outside air) and outside air.
  The interior gets an ignore label: Minecraft builds are hollow and a photo never shows
  the inside. Interiors are a deterministic post-process (hollow, optional floor plates).
- **Frame:** Sponge (Y, Z, X) arrays; 3 blocks of supervised air around the sides and top.
  The degraded input is voxelised into the **same** frame, voxel for voxel.
- **Scale:** from the caption's storey count when present, else the file's units if they
  read plausibly as metres, else a category prior; 3–6 blocks per storey (3–4 is roughly
  real scale, 5–6 is what the library's builders use). Pilot sources: 61 caption, 36 file
  units, 69 priors.
- **Neighbour-dependent states** (stair corners) are derived after assembly with the
  game's own rule, never predicted.

### 5.2 Buildings larger than the window

Inference runs overlapping windows (`mcvoxel.tiling`): stride = window − overlap, last
window snapped to the grid's end, each axis shorter than the window padded. Shape
probabilities are blended with weights that ramp down across the overlap (flat at grid
edges); material comes from the window that sees the voxel most centrally, so no 120-channel
material accumulator is needed over a large grid. A test feeds deliberately corrupted predictions in a
4-voxel band at every window edge and recovers the ground truth exactly. Memory for the
shape accumulator is about 400 MB for a 256 × 384 × 256 box.

Two limits to respect: Minecraft 1.18+ builds within 384 blocks of height, so the pilot
caps height at 300 blocks; the pilot also caps footprint at 220 blocks (time, not memory).

### 5.3 GPU benchmark (RTX 4060 Ti 16 GB)

A throwaway 3D residual U-Net (7.2M parameters, 6 input channels, 8 + 120 outputs), bf16,
forward + backward + AdamW:

| Window | Batch | Checkpointing | Peak VRAM | Seconds / step |
| --- | --- | --- | --- | --- |
| 64³ | 8 | no | 7.7 GB | 0.42 |
| 64³ | 16 | yes | 9.3 GB | 1.07 |
| 96³ | 4 | no | 13.0 GB | 0.80 |
| 96³ | 4 | yes | 7.9 GB | 0.84 |
| 128³ | 1 | no | 7.7 GB | 0.42 |
| 128³ | 2 | yes | 9.3 GB | 1.12 |

Throughput is 4–5 million voxels per second at every size, so larger windows cost nothing
per voxel; they buy context at the price of a smaller batch. Runs that peak above about
15 GB spill into shared system memory and are 3–65× slower (64³ at batch 16 without
checkpointing: 2.8 s per step instead of about 0.85). spconv 2.3.6 already works on this
machine inside the TRELLIS environment, so a sparse model is an option later.

---

## 6. The pilot

*Historical: Objaverse pilots on a path parked by D13 ([version 4](#version-4-2026-10-11-schematic-first-pivot)).*

### 6.1 What ran

200 objects chosen by `synth.select_pilot` with fixed quotas (detached houses 26, of which
suburban 6; apartments 20; mid-rise 16; skyscrapers 20; offices 8; commercial 18;
institutional 14; religious 18; towers 18; fortifications 12; industrial 10; townhouses 8;
pavilions 6; generic 6), CC0 or CC BY only, at most two per author, at most 15% scans.
Each object: download, orient, scale, voxelise the target, assign blocks, write and
round-trip the `.schem`, degrade, voxelise the input in the same frame, record metrics.

| Outcome | Objects |
| --- | --- |
| **Pairs written** | **166** |
| Rejected: thin shell (open or paper-thin model) | 15 |
| Rejected: flat site | 7 |
| Rejected: scattered scene | 5 |
| Rejected: already blocky source | 4 |
| Rejected: fragmented | 3 |

| Per pair | p10 | p50 | p90 | max |
| --- | --- | --- | --- | --- |
| Height, blocks | 12 | 26 | 77 | 190 |
| Footprint, blocks | 17 | 36 | 89 | 212 |
| Shell voxels | 262 | 1,631 | 11,010 | 51,598 |
| Seconds (6 workers in parallel) | 5.2 | 10.1 | 28.2 | 53.7 |
| KB on disk | 101 | 166 | 275 | 586 |
| Input/target occupancy IoU | 0.59 | 0.79 | 0.88 | 0.96 |

Glass: 67 sources had glass materials, 54 targets kept glass blocks.

### 6.2 Hand audit

I graded all 195 pre-gate targets from contact sheets (`visual_audit.json`); the gates then
removed 14 junk, 8 flawed and 7 clean ones — 4 of those "clean" were blocky sources whose
targets looked fine but whose inputs would break Decision 1's premise.

| Category | Pairs | Clean | Flawed | Junk |
| --- | --- | --- | --- | --- |
| Skyscraper | 17 | 15 | 2 | 0 |
| Office | 6 | 6 | 0 | 0 |
| Mid-rise | 15 | 11 | 4 | 0 |
| Apartment | 19 | 12 | 6 | 1 |
| Institutional | 10 | 7 | 2 | 1 |
| Pavilion | 5 | 4 | 1 | 0 |
| Religious | 17 | 11 | 2 | 4 |
| Tower | 16 | 8 | 2 | 6 |
| Fortification | 9 | 3 | 2 | 4 |
| Industrial | 7 | 3 | 2 | 2 |
| Commercial | 12 | 2 | 2 | 8 |
| Houses (4 subtypes) | 20 | 7 | 7 | 6 |
| Townhouse | 7 | 3 | 2 | 2 |
| Generic | 6 | 2 | 3 | 1 |
| **All** | **166** | **94 (57%)** | **37 (22%)** | **35 (21%)** |

What is left in the junk: small detailed objects (shops, sheds) drawn at 10–16 blocks
become blobs; fantasy objects the text filters miss; roofless "dollhouse" models; scene
clutter such as fenced plots.

### 6.3 First measured gap: stairs

| | Stairs | Slabs | Full |
| --- | --- | --- | --- |
| Pilot targets (supervised shell) | 1.5% | 19.6% | 78.9% |
| Real builds (structural blocks) | 16.3% | 10.6% | ~60% |

Geometry-derived targets use stairs only where a surface actually crosses a voxel as a
45°-ish step; builders put stair courses on roofs of almost any pitch, and cover flat roofs
with full blocks rather than half-block slabs. A model trained on these targets will
under-use stairs. Decision D1.

### 6.4 Automatic QA gates, evaluated against the audit

Each gate on its own, over the 195 audited targets (49 junk, 45 flawed, 101 clean):

| Gate | Junk caught | Flawed caught | Clean lost | Adopted |
| --- | --- | --- | --- | --- |
| Thin shell (interior share < 0.15) | 11 | 3 | 1 | yes |
| Flat site (height / footprint < 0.2) | 5 | 4 | 1 | yes |
| Fragmented (largest part < 85%) | 3 | 2 | 2 | yes |
| Already blocky (vertex lattice ≥ 0.5) | 0 | 0 | 4 (blocky sources) | yes |
| Roofless (roof coverage < 0.5) | 33 | 22 | **31** | no |
| Too small (footprint < 16) | 16 | 12 | **16** | no |

Applied together in the v0.3 run (a pair is rejected at its first failed gate), the adopted
four removed 14 junk, 8 flawed and 7 clean targets, 4 of the clean ones being blocky
sources.

The blocky detector (`synth.blockiness`) catches what text cannot: a Minecraft-built
"Cathedral" passed every text rule. Ordinary sources score ≤ 0.023 at the 97th
percentile; the four flagged score 0.66–1.0. Remaining junk needs a different signal — an
image classifier on the Sketchfab thumbnails is the obvious one (Decision D2).

### 6.5 Cost of the full run, extrapolated

| | Pilot | 15,000 objects |
| --- | --- | --- |
| Wall time (6 workers) | ~8 min | ~12 h |
| Pairs on disk (with previews) | 37 MB | ~3 GB |
| Downloads (streamed, deleted after) | 2.0 GB | ~150 GB transient |

---

### 6.6 Pilot v2 (2026-09-25): the V1 pipeline

200 assets, same quota idea with civic split out (residential 54, of which suburban houses 6),
photogrammetry excluded, 1:1 scale, builder roofs, the fixed voxeliser. 185 of the 200 GLBs
were already on disk from pilot v1 (only 113 MB downloaded); about 8 minutes on 6 workers.
Full numbers: `data/synthetic/pilot-v2/report.md`.

| Outcome | Assets |
| --- | ---: |
| **Pairs written** | **120** |
| too_small at 1:1 (footprint and height both under 16) | 44 |
| thin_shell | 11 |
| scan_geometry (dense mesh, no planar faces) | 7 |
| blocky_geometry (already voxels) | 5 |
| scattered / fragmented / flat_site | 4 / 3 / 3 |
| too_tall (> 300 blocks) / too_large | 2 / 1 |

* **Diversity held.** Pairs by group: residential 25% (detached houses 9%), high-rise 12%,
  religious 10%, mid-rise 10%, commercial, tower and fortification 8% each, institutional 7%,
  industrial 4%, civic 3%. The losses to `too_small` fall mostly on commercial (8 of 24),
  industrial (6 of 12) and residential (14 of 54): at 1:1 many kiosks, sheds and cottages are
  under 16 blocks. The tall exception kept 9 slender buildings (towers, a lighthouse, a water
  tower, a narrow office block).
* **Size at 1:1.** Footprint p10/p50/p90 17 / 31 / 92 blocks (max 186); height 10 / 21 / 107
  (max 260). 72% of pairs fit one 64^3 window; the largest needs 16 tiles.
* **Scale sources.** File units 85 (23 of them agreeing with caption storeys), caption storeys
  17, category prior 18. Confidence: high 23, medium 56, low 41.
* **Roofs.** On evenly pitched roof columns, the share capped with stairs: real builds 96.7%;
  builder targets 50.1% on houses (33.7% overall); geometric targets 25.9% on houses (8.2%
  overall). Builder stairs face up-slope 93–98% of the time. The rest of the gap is mostly
  columns the roof pass leaves alone (thin edges, glass) and non-roof steps (setbacks).
* **Visual audit** (mine, from the contact sheets, approximate): about 70 good, 30 flawed,
  20 garbage of 120 (58 / 25 / 17%). Garbage causes: buildings on raised or round plots
  (islands, beaches, moats, desert discs) that ground-plate removal cannot strip; roofless,
  cutaway and floor-plan models; a misclassified object (a cinema armchair) and a flagpole;
  one cartoon church. Flaws: gardens, pools and trees kept; dark curtain walls without glass;
  thatch roofs taking the colour of beams; about five scale errors (landmarks, a 17 m
  cathedral, a 107 m village church).
* **Cost.** 7.5 s per asset (p50), training files 14 KB per pair; review renders are 94% of
  the 100 MB (configurable: `cfg.renders`).

## 7. Degrading the input like TRELLIS

The input is not arbitrary noise. Each step reproduces a known TRELLIS property
(`synth/degrade.py`):

- **Resolution bottleneck** — TRELLIS works in a 64³ structure normalised to the object's
  longest axis, so detail is lost *relative to size*. The clean model is voxelised at
  96–160 cells across, blurred and re-extracted with marching cubes.
- **Surface noise** ("melted" buildings), **holes and floaters**.
- **Baked colour** — darker, lit from a random sun, hue-drifted, noisy.
- **Hallucinated unseen side** — surfaces facing away from a random camera blend toward the
  mirrored visible side, darker, sometimes tinted wrong.

Calibration against the 54 cached TRELLIS meshes, each voxelised at 32, 48 and 64 blocks
(162 samples), with the pilot inputs voxelised the same way (p10 / p50 / p90; v0.1 is the
195 pre-gate inputs, v0.2 the 166 current ones):

| Statistic | TRELLIS | Pilot v0.1 | Pilot v0.2 (current) |
| --- | --- | --- | --- |
| Fill gain | 1.5 / 2.7 / 4.5 | 1.1 / 2.1 / 4.4 | 1.2 / 2.1 / 4.6 |
| Blobby shell (3+ open faces) | .03 / .07 / .17 | .05 / .12 / .22 | .04 / .10 / .20 |
| Brightness L* | 10 / 22 / 29 | 12 / 25 / 42 | 9 / 19 / 33 |
| Colourfulness C* | 3.3 / 6.8 / 11.6 | 1.4 / 7.2 / 19.6 | 1.2 / 4.5 / 13.5 |
| Neighbour ΔE | 3.1 / 4.9 / 9.1 | 1.8 / 4.3 / 8.6 | 1.5 / 3.4 / 6.3 |

v0.2 tightened brightness, saturation and roughness; median colourfulness and colour noise
are now slightly *below* TRELLIS. The reference set is small (54 meshes, not all
buildings), which is one more reason for the real-TRELLIS calibration subset below.

---

## 8. Model and training (unchanged in outline; nothing implemented)

*Splits and the never-train-on-real rule superseded by D13 and D17, input resolution by D16 ([version 4](#version-4-2026-10-11-schematic-first-pivot)).*

- Dense 3D residual U-Net, 64³ windows to start (batch 8), 96³ as the first upgrade; two
  heads (shape, material); inputs: occupancy, surface flag, RGB, height.
- Loss: class-weighted shape cross-entropy, material cross-entropy on supervised solid
  voxels, ignore label on interiors, material loss masked for geometry-only (LoD2)
  samples.
- Augmentation: 90° rotations and mirrors through `blockstates.rotate_y` / `mirror_x`, so
  stair facings stay correct.
- Material consistency across windows on large buildings: a global colour descriptor as a
  conditioning input, plus region-level material voting after assembly.
- Splits by source author. The real set is never trained on (section 4.3).
- Recommended before training: 300–500 pilot sources rendered and run through real TRELLIS
  (~10 GPU-hours) as a validation set of genuine TRELLIS inputs with clean targets.

---

## 9. Decisions (V1, answered 2026-09-25)

| # | Question | V1 decision | Where it lives |
| --- | --- | --- | --- |
| D1 | Roof targets | Builder-style roofs (Approach B) are the primary target; the geometric conversion (A) is stored beside it for comparison | `mcvoxel.roofs`, `cfg.roof_style`, `cfg.store_geometric` |
| D2 | Junk | Deterministic metadata and geometry filters first, every rejection recorded; a learned classifier only as a hook | `synth.filters` (`CLASSIFIERS` empty: TODO) |
| D3 | Scans | Photogrammetry excluded from V1 | `cfg.exclude_photogrammetry`: text + Objaverse++ + a geometric scan check |
| D4 | Minimum size | ~16 blocks of footprint after 1:1 scaling, configurable; small assets rejected, not rescaled | `cfg.min_footprint_blocks` (+ tall exception, see below) |
| D5 | Scale | **Superseded by D15 ([version 4](#version-4-2026-10-11-schematic-first-pivot)) for the schematic dataset.** 1:1 with the player; real proportions; bounded against bad metadata | `synth.scale`, `cfg.blocks_per_metre = 1.0` |
| D6 | Licences | **Superseded by D14 ([version 4](#version-4-2026-10-11-schematic-first-pivot)) for the schematic dataset.** Only clearly permissive (CC0, CC BY), recorded per pair | `cfg.licences`, manifests, .schem Metadata |
| D7 | Berlin LoD2 | Not a V1 dependency; documented plug-in point | dataset-pipeline.md, "Adding a data source" |
| D8 | Storage | ~150 GB temporary; stream: download, process, validate, save, delete | `cfg.tmp_dir`, `cfg.keep_raw`, `synth.run` |
| D9 | Real evaluation set | **Superseded by D17 ([version 4](#version-4-2026-10-11-schematic-first-pivot)).** ~64 diverse builds, held out | `eval/real-eval-v1.json`, `synth.real_eval.assert_disjoint` |
| D10 | `mcvoxel` | Shared package, only as clean engineering (API, tests, docs, no app hacks) | Real2Minecraft `mcvoxel/` (+ README) |
| D11 | Voxeliser bug | Fixed at the root, tests, used everywhere | `mcvoxel.raster.clean_solid` |
| D12 | Height | 300-block bound, no compression of normal buildings | `cfg.max_height_blocks` (too_tall rejects) |

Still open (details in the pilot v2 report and the final summary):

* **Tall exception to the footprint rule** — added so towers, lighthouses and minarets are
  not systematically rejected (`cfg.min_tall_height_blocks = 16`; 0 = footprint only). It
  kept 9 pilot pairs, mostly good ones. Needs your confirmation.
* **Low-pitch roof step** — builders use both a full block and a top slab; `RoofParams.low_upper`.
* **Plots and scenes** — the largest remaining junk cause (round islands, gardens, beaches).
* **Scale of landmarks** — assets rarely carry true size; priors give plausible, not true, heights.

---

## 10. Next steps, in order

*Superseded by the [version 4 phase plan](#phase-plan).*

1. Your review of pilot v2 in the review tool (`python -m synth.review data/synthetic/pilot-v2`).
2. Real TRELLIS vs synthetic degradation: input images for the 120 pilot pairs are rendered;
   start the TRELLIS server, run `trellis_compare generate` then `compare`, retune `cfg.degrade`.
   Extend to 300–500 buildings from a larger run.
3. Plot/scene handling and a thumbnail classifier for the remaining junk.
4. A scaled run (1–2k assets, `renders="review"`), then the full streamed run.
5. Model and training loop at 64³ on builder targets; evaluation on held-out synthetic pairs.
6. Real evaluation pairs (AI photo + TRELLIS for the 64 held-out builds) and the gap report.
7. Berlin LoD2 reader (phase 2).

---

## 11. How to run it

*These commands run the Objaverse path, parked by D13 ([version 4](#version-4-2026-10-11-schematic-first-pivot)).*

See [dataset-pipeline.md](dataset-pipeline.md#running-it). In short:

```powershell
.venv\Scripts\python -m pip install -r requirements-synth.txt
.venv\Scripts\python -m synth.select --config data/synthetic/pilot-v2/run_config.json
.venv\Scripts\python -m synth.run    --config data/synthetic/pilot-v2/run_config.json --workers 6
.venv\Scripts\python -m synth.report data/synthetic/pilot-v2
.venv\Scripts\python -m synth.review data/synthetic/pilot-v2
.venv\Scripts\python -m unittest discover -s tests                      # 339 tests
```
