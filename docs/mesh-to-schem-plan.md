# Mesh → .schem: dataset, voxelizer, model and eval plan

*Proposal, 2026-09-12. Research and design only — no model code, no pipeline code, nothing
in the Flask app touched. Section 5 lists the decisions this needs from you before anything
is built.*

The pivot decouples two problems that the manual collector and `house_grammar` were trying
to solve at once: **where building shapes come from** (a bulk 3D dataset of real
architecture) and **how a shape becomes a Minecraft build** (deterministic voxelize +
palette code, then a learned model that does it robustly on TRELLIS output). This document
proposes how to do each half so it fits what already exists in this repo and in
`Real2Minecraft`.

---

## 0. Four findings that change the plan

**1. The fine-tune / eval pairs do not exist yet.** All 122 builds in the library are at
`needs_image`: there are 110 reference screenshots, **zero** AI-generated images and
**zero** TRELLIS meshes (`data/collector.db` and `git ls-files` agree). The 106 real
schematics are usable as *targets* today, but every real `(mesh, schem)` pair still has to
be produced by hand through the screenshot → AI photo → TRELLIS path. Section 4 budgets
that.

**2. Most of "step 2" already exists — in `Real2Minecraft`, not here.** The request calls
colour/material → blockstate "new". It is not: the sibling project
`C:\Users\mendo\Documents\Projects\Real2Minecraft` (last commit 2026-07-14) already has

| Module | What it does |
| --- | --- |
| `voxelize.py` | SAT triangle-box surface rasterisation (gap-free), exterior flood fill, cleanup — the orchestrator `voxelize_mesh` |
| `voxel_sampling.py` | per-voxel colour from the mesh texture: mode-seeking, side-aware surface splatting |
| `color_quantize.py` | k-means + merge in weighted LAB, shadow → albedo repaint, spatial coherence |
| `block_palette.py` | ~113-block curated palette, weighted-CIELAB KD-tree `match_blocks`, `AUX_PALETTE` for glass, `match_blocks_restricted` |
| `block_shapes.py` | octant-based full / slab / stairs classification with facing, `BLOCK_VARIANTS` slab+stair table, `apply_shapes` |
| `mesh_normalize.py`, `building/yaw_align.py` | Y-up ground placement, XZ centring, facade-to-axis yaw alignment |
| `building/massing.py`, `facade*.py`, `assemble.py` | rule-based building mode with per-voxel semantic labels (wall / window / door / roof / floor …) |
| `trellis_client.py` | drives the local TRELLIS FastAPI server (`:7960`, simplify 0.9, 1024 texture, 24 steps) |
| `training/__init__.py` | a stub for exactly the "degrade a mesh to imitate TRELLIS" idea |
| `outputs/trellis/` | **228 cached real TRELLIS outputs (358 MB)** — the calibration set for any "make it look like TRELLIS" step |

What is genuinely new is: bulk ingest of a 3D dataset, running that code offline over
thousands of meshes, the **input/target asymmetry** below, the pair format, and the model.

**3. A deterministic target is only worth learning if the input is worse than what made
the target.** If the target `.schem` is `f(mesh)` and the model's input is the same mesh,
the model learns to imitate `f` — which you could just run at inference. The model earns
its place only if the training input is a *degraded, TRELLIS-like* version of the source
mesh while the target is computed from the *clean* source (plus information the clean
source has and TRELLIS output does not: material names, transparency flags, exact
geometry). This is Decision 1 and it shapes the whole data generator.

**4. Your compute is a single 16 GB consumer GPU.** RTX 4060 Ti 16 GB, 32 GB RAM,
i9-12900K, **164 GB free on C:**. That is enough to train the model proposed in section 3
locally in days, but not enough disk to keep the raw meshes of a 15k-object subset
(≈165 GB at Objaverse's ~11 MB average): the ingest has to stream (download → voxelize →
delete).

---

## 1. Source datasets

### 1.1 Candidates compared

Counts marked **measured** come from files I downloaded and scanned during this research
(one 5,000-object Objaverse metadata shard, the LVIS annotation file, and both
TRELLIS-500K Objaverse csvs); everything else is from the dataset's own documentation.

| Dataset | What it is | Houses / towers | Texture & quality | Usable count (est.) | Licence | Bulk download + filtering |
| --- | --- | --- | --- | --- | --- | --- |
| **Objaverse 1.0 (Sketchfab)** via TRELLIS-500K + Cap3D captions | 798k CC-licensed artist models and photogrammetry scans | **measured:** 17.6% of TRELLIS-500K's 168,307 Sketchfab objects have a building word in their caption (29,574); 13,959 are house-like and not blocky. In the raw shard: 9.0% carry Sketchfab's `architecture` category; 1.6% pass keyword ∧ architecture ∧ not-Minecraft → ≈12k over the full set | Textured (UV atlas or vertex colour). Mixed: face count p10 374 / median 18k / p90 916k. ~3–4% Minecraft/LEGO/voxel contamination in the building hits — must be filtered out | **8–15k** standalone buildings after filters | Per object. **Measured shard:** CC-BY 88.4%, BY-NC-SA 5.8%, BY-NC 3.1%, BY-SA 2.2%, CC0 0.5%. Dataset wrapper ODC-By 1.0 | Excellent. `objaverse.load_objects(uids)` downloads by uid; captions (Cap3D 255 MB csv, TRELLIS-500K 167 MB csv with aesthetic score), Sketchfab tags/categories, Objaverse++ quality flags all keyed by uid |
| **Objaverse-XL GitHub subset** (TRELLIS-500K `ObjaverseXL_github.csv`) | 3D files scraped from public GitHub repos | **measured:** 22,696 of 311,843 building-captioned; 7,706 house-like non-blocky (many are game assets, OSM and swisstopo exports) | Lower and more uniform quality | ~5–8k | **No per-object licence.** The Objaverse-XL paper states users must assess each object's licence for downstream use; most repos carry none | Same toolkit, but licence triage is manual |
| **BuildingNet** | 2,000 textured building meshes mined from Trimble 3D Warehouse, 513k labelled parts (wall / roof / window / door …) | Residential 1,424, Religious 540, Commercial 153, Castles 85, Civic 67 | High, curated, standalone by construction, part labels | 2,000 (1,424 houses) | Access by Google form; no licence text found on the site or in the paper. Underlying 3D Warehouse ToS forbids "aggregation" and redistribution → treat as **research / eval only** | One zip after approval |
| **PLATEAU (Japan)** | Government LoD2 city models, ~210 municipalities, subset textured from oblique aerial imagery | Millions of real buildings, all types; Tokyo LoD2 textured in places | Real roof geometry, real facade colours at aerial-texture resolution; **no window/door geometry** | Very large; needs splitting per building (CityGML already does) | **CC BY 4.0**, commercial use allowed | CityGML per municipality (also OBJ/FBX); textured subsets are separate downloads |
| **Berlin 3D** | City-wide textured LoD2 CityGML | ~all Berlin buildings | Textured LoD2, no openings | large | dl-de/zero-2.0 per GovData listing | Per-district CityGML + texture packs |
| **Helsinki 3D** | Textured LoD2 CityGML + photogrammetric reality mesh | city | Textured LoD2 | large | CC BY 4.0 | CityGML per 4 km² sheet, OBJ |
| **3D BAG (Netherlands)** | LoD2.2 auto-reconstructed from LiDAR + footprints | ~10M buildings, mostly terraced | **Untextured** massing with real roof shapes | 10M | CC BY 4.0 | CityJSON / OBJ / GPKG tiles |
| **Building3D (Estonia)** | LiDAR + roof wireframes + meshes, 160k buildings, 16 cities | many | Untextured roofs/massing | 160k | **CC BY-NC-SA 4.0** → non-commercial | website |
| **BuildingWorld (AAAI 2026)** | 5M LoD2 models from 44 cities on five continents | many | Untextured | 5M | Not stated; public release not confirmed | Not confirmed |
| **ArchiSet (ICCV 2025)** | 13,728 "real-world architectural design" building forms as point clouds / voxels / meshes + window-to-wall ratio | design massing | Untextured massing + window masks | 13.7k | Not located; no download page found | Not located |
| **ShapeNet** | 55 object categories | **none** — ShapeNetCore has no building category | — | 0 | — | — |
| **3D-FRONT** | 18,968 furnished *indoor* rooms, 13k furniture models | **none** — no facades, roofs or exteriors | — | 0 | — | — |

Two candidates from the request are therefore out: ShapeNet has no buildings at all, and
3D-FRONT is interiors only. Objaverse's LVIS labels are also useless for this — the
1,156 LVIS categories contain no `house` or `building` class (the nearest are `windmill`
112, `dollhouse` 56, `birdhouse` 50, `clock_tower` 49), so the filter has to be
caption/tag based, which is what the measured numbers above use.

### 1.2 Recommendation

**Primary: Objaverse 1.0 Sketchfab objects, filtered through TRELLIS-500K.** Three
reasons beyond count and licence:

- **It is the distribution TRELLIS was trained on.** TRELLIS-500K is Microsoft's own
  curated training list (Objaverse-XL Sketchfab + GitHub, ABO, 3D-FUTURE, HSSD, Toys4k,
  filtered by aesthetic score). Buildings that TRELLIS learned from are the buildings it
  reconstructs best, so a model trained on them sees the same kind of geometry at
  inference.
- **Per-object licence in metadata**, so a licence policy is a filter, not a research job.
- **Real architecture, real materials.** The captions read like the brief: "two-story house
  with a light tan gable roof, light brown walls with dark brown beams, white-framed
  windows, a verandah…". Photogrammetry scans of real houses are in there too.

**Filter recipe** (all deterministic, all recorded per object in the dataset index):

1. Start from `ObjaverseXL_sketchfab.csv` (TRELLIS-500K): has `sha256`,
   `file_identifier`, `aesthetic_score`, `captions`.
2. Caption regex for building nouns (house, cottage, cabin, villa, townhouse, apartment,
   church, castle, tower, barn, lighthouse, …) **and** Sketchfab category `architecture`
   or a building tag from the Objaverse 1.0 metadata shards.
3. Reject if caption/name/tags match `minecraft|voxel|blocky|lego|pixel|low.?poly|isometric`
   — the dataset must not contain things that are already block-shaped.
4. Licence ∈ {`cc0`, `by`} (add `by-sa` if a share-alike derivative dataset is acceptable;
   exclude `by-nc*` unless the product is definitely non-commercial — Decision 2).
5. Objaverse++ flags (ODC-By, on Hugging Face): quality ≥ Medium, `not_single_object` = no,
   `scene` = no, `transparent` = no.
6. Geometry sanity after download: 1k ≤ faces ≤ 2M (decimate above 150k), bounding box
   aspect within the library's range (footprint aspect p10 1.06 / p90 1.83; height ≤ 3×
   min footprint side for houses, allow more for towers), `yaw_align.estimate_yaw`
   wall-area fraction ≥ 0.2 (a building has walls), one dominant connected component
   after voxelisation (≥ 80% of solid volume, the same idea as `house_stats`'s
   `MIN_COMPONENT_SHARE`), and no ground plane larger than the building footprint (crop
   plots off with the same "trim to the build" logic as `VoxelWorld.auto_bbox`).
7. Keep an index row per object: uid, sha256, licence, author, aesthetic score, caption,
   filter outcomes, generation seeds. That is the dataset's provenance and its ablation
   handle.

Expected yield: **8–15k buildings**, of which roughly half are houses. Raw download
≈ 100–165 GB streamed; retained voxel pairs ≈ 0.3–0.6 MB each → **5–10 GB on disk**.

**Secondary (phase 2, optional): textured city LoD2 (PLATEAU, Berlin, Helsinki).** Real
roof geometry at real-world scale in metres for millions of ordinary buildings, and real
facade colours — but no window or door geometry, so their targets need the
"openings from texture" rule in 2.3 and they should be a separate domain flag in the index.
Their absolute scale is the one thing Objaverse lacks, which makes them the right place to
calibrate blocks-per-metre later.

**Eval only: BuildingNet**, if the access form is granted and the project stays
non-commercial. Its part labels (window, door, roof) are the only ground truth available
for checking the palette rules, but its 3D Warehouse origin makes it unsafe as training
data for anything you might ship.

**Not recommended:** Objaverse-XL GitHub subset (no licences), Building3D (NC-SA),
BuildingWorld and ArchiSet (not verifiably available, untextured).

---

## 2. Voxeliser and palette design

### 2.1 Frames and scale

- **Axes.** TRELLIS exports Y-up (`to_glb` rotates z-up → y-up). `Real2Minecraft` grids are
  `(W, H, L)` = `(x, y, z)`; this repo's `schem_voxels.VoxelGrid`, `schem_write` and
  `VoxelWorld.extract_region` use Sponge order `(H, L, W)` = `(y, z, x)`. One transpose,
  `grid_yzx = grid_xyz.transpose(1, 2, 0)`, and every tool here (thumbnails, Inspect,
  `contact_sheet`, `house_stats`) works on the generated targets unchanged.
- **Scale.** Objaverse meshes have no trustworthy units and TRELLIS output is
  unit-normalised, so blocks-per-metre cannot be measured on the input. Instead **sample
  the target size from the real library**: longest horizontal axis drawn from the measured
  footprint distribution (p10 16 / median 26 / p90 46 blocks, max 92) with the same
  clipping `house_params` applies, and generate each source mesh at **two or three sizes**
  so the model learns scale invariance. At inference the size is the user's slider, as in
  `Real2Minecraft` (`VOXEL_MIN_BLOCKS` 16 … `VOXEL_MAX_BLOCKS` 128). Decision 3 fixes the
  envelope.
- **Alignment.** `mesh_normalize.normalize_mesh` (ground at y=0, XZ centre) then
  `yaw_align.estimate_yaw` so facades sit on the grid — the same two calls the inference
  path already makes, which is what keeps training and inference in one frame.

### 2.2 Geometry: mesh → solid voxels → shapes (existing code, three changes)

The pipeline is `Real2Minecraft`'s, run offline:

```
glb ──► normalize + yaw ──► scale longest axis to N blocks
    ──► _voxelize_surface_sat (gap-free surface)
    ──► exterior flood fill + _clean_voxels (solid, leak-guarded)
    ──► classify_shapes on a 4x finer grid ──► full / slab(top,bottom) / stairs(4 facings)
```

Changes needed for bulk use:

1. **Pre-decimate.** `_voxelize_surface_sat` loops over triangles in Python; at the
   Objaverse median (18k faces) it is seconds, at p90 (916k) it is minutes. Decimate to
   ≤150k faces before SAT (trimesh's quadric decimation needs the `fast-simplification`
   package). Budget: ~30 s per mesh per size → 15k meshes × 2 sizes ≈ 10 CPU-days single
   core, ≈ 1.5 days on the 12900K's 8 P-cores.
2. **Leak policy.** `classify_shapes` returns `None` when the fine flood leaks (non-
   watertight mesh) and everything becomes full cubes. For a *training target* that is a
   corrupt label, not a fallback: mark the sample `leaked` in the index and drop it.
3. **Plot removal.** Many Sketchfab houses sit on a lawn/plinth slab. Detect a flat
   bottom layer whose footprint exceeds the walls' footprint by > 15% and cut it, the way
   `house_stats` finds the pad; record `had_plot`.

### 2.3 Colour and material → blockstate (existing matching + new privileged rules)

The existing chain is `_sample_voxel_colors_texture` → `_quantize_colors` →
`match_blocks` (weighted LAB, `L_WEIGHT` 0.65) → `apply_shapes` (`BLOCK_VARIANTS` for the
slab/stair variant, nearest variant-capable material otherwise). Keep it as the *default*
assignment. Add rules that use information only the clean source mesh has — this is what
makes the target richer than anything computable from the TRELLIS input:

| Rule | Signal on the clean mesh | Target |
| --- | --- | --- |
| Windows | glTF material `alphaMode` BLEND/MASK, `baseColorFactor` alpha < 1, or material/mesh name matching `glass|window|pane` | `AUX_PALETTE` glass (colour-matched), full cube in v1 |
| Doors | material/mesh name `door`; else a ground-touching opening 1–2 wide, 2–3 tall, in a wall cell | `minecraft:<wood>_door[half=…]` (v2), wood plank cube (v1) |
| Roof vs wall | face normal `|n_y|` > 0.35 and above the top storey line (the `voxel_sampling` UP/SIDE buckets already exist) | roof voxels are matched against the *stair-capable* subset so slopes always become stairs/slabs of one material; walls against the full palette |
| Trim / frames | a thin (≤ 1 voxel) colour region that outlines a window or a corner | keep as its own material (do **not** let `_quantize_colors` absorb it) — this reproduces the accent behaviour `house_grammar` had to fake |
| Ground contact | the bottom course | matched against stone/brick families (foundation), mirroring `house_stats`'s "foundation" material |

Two consistency items with this repo:

- `block_roles.stairs_for/slab_for` and `Real2Minecraft.block_shapes.BLOCK_VARIANTS` are
  the same table written twice (and `block_roles` also knows `pale_oak`, tuff bricks and
  deepslate tiles). Unify into one and have both import it.
- `block_colors.py` (thumbnail colours) and `block_palette.BLOCK_PALETTE` (matching
  colours) disagree by a few units per block (stone 125 vs 126). Harmless for thumbnails;
  for matching, derive `BLOCK_PALETTE` from real texture averages once (a resource pack
  dropped into `Real2Minecraft/textures/blocks/` is already supported by
  `block_textures.get_tile`).

**Vocabulary.** Measured over the 106 real builds (1.53M non-air blocks): 4,228 distinct
blockstates and 795 base blocks, but the top-100 base blocks cover 88% of volume and the
top-200 cover 96%. Of the *structural* blocks (terrain and plants removed): 60% full
cubes, 16% stairs, 11% slabs, 13% other non-cube (panes, fences, doors, trapdoors). So the
target vocabulary is **material × shape**, factored:

- **materials**: ~128 = `BLOCK_PALETTE` (113) + `AUX_PALETTE` glass (9) + a handful the
  library uses heavily that the palette lacks (`quartz_pillar`, `bone_block`,
  `white_concrete_powder`, `stripped_*_log`) — final list chosen by library frequency,
  frozen in `vocab.json`.
- **shapes**: 8 = air, full, slab-bottom, slab-top, stairs N/E/S/W (v1). Panes, fences,
  doors, trapdoors are v2 (Decision 4).

### 2.4 The input side: making the source mesh look like TRELLIS output (Decision 1)

Calibrate every parameter below against the 228 cached real TRELLIS outputs in
`Real2Minecraft/outputs/trellis/` (face count after 0.9 simplify, curvature, texture
frequency content, seam and hallucination statistics). Apply to the *clean* mesh, in this
order, with a per-sample seed recorded in the manifest:

1. **Remesh + simplify** to a TRELLIS-like face budget (uniform remesh, then decimate to
   5–30k faces), so sharp corners round off the way they do in TRELLIS.
2. **Taubin smoothing** (1–3 iterations) — TRELLIS meshes are blobby; measure the amount.
3. **Thin-feature loss**: drop or fatten geometry thinner than ~0.6 voxel at the target
   size (railings, mullions, eaves edges), which TRELLIS reliably loses.
4. **Texture re-bake at 1024²** with low-pass filtering, a global tone shift (TRELLIS bakes
   are darker), baked-lighting gradients (top bright, underside dark — exactly what
   `color_quantize`'s shadow stage repairs), and seam noise.
5. **Unseen-side hallucination**: pick a random photo azimuth; faces facing away get
   blurred, desaturated and partly recoloured from a *mirror* of the seen side — this is
   what `color_repair` fights at inference and the model should learn to see through it.
6. **Pose jitter**: yaw ±3° (yaw_align residual), scale ±5%, XZ offset ±0.5 block.
7. Then run the **same** `normalize → yaw → SAT → flood` on the degraded mesh at the same
   target size to produce the input tensors.

**Optional but recommended calibration subset:** render 300–500 source meshes to a
photo-like image (Blender/Cycles with a sky HDRI, or the same AI-photo step the manual
pipeline uses) and push them through the local TRELLIS server with
`trellis_client._run_trellis_api`. At 1–2 min per run on the 4060 Ti that is ~10 GPU
hours, and it gives real TRELLIS inputs paired with clean targets: a validation set that
tells you whether the synthetic degradation is realistic, and a way to tune it. Doing this
for the *whole* dataset (15k × 1.5 min ≈ 16 GPU-days) is possible but should wait until
the synthetic degradation is shown insufficient.

### 2.5 Writing a pair

- **Target**: `schem_write.write_schem(grid_yzx, names, path, metadata=…)` from this repo.
  It is deterministic (sorted palette, `mtime=0`), re-densifies the palette, and stamps
  provenance into `Metadata` — so every generated `.schem` carries `source_dataset`, `uid`,
  `sha256`, `licence`, `author`, `target_size`, `degrade_seed`, `pipeline_version`, and
  can be dropped into the library's importer unchanged for inspection.
- **Input**: `input.npz` — `occ` uint8 `(H,L,W)`, `rgb` uint8 `(3,H,L,W)`, `normal` int8
  `(3,H,L,W)` (surface voxels only; optional), plus `origin`, `scale`, `yaw_deg`.
- **Target cache**: `target.npz` — `shape` uint8, `material` uint16 (indices into
  `vocab.json`), derived from the `.schem` by `schem_voxels.load_voxels` + a vocab mapper,
  so the `.schem` stays the single canonical target.
- `manifest.json` per sample (index row + timings + QA results), `preview.png` from
  `thumbs.render_schem` next to the input rendered with `thumbs.render_mesh` — the same
  shared camera the Library uses to spot mismatched pairs at a glance.

### 2.6 QA on every sample (reuse `gen_houses.validate`)

`schem_convert.inspect` on the written file, decode it back, compare block-for-block;
then the geometry gate: solid fraction inside the library's measured density range
(0.04–0.55), a roof covering ≥ 80% of the footprint, stair share of the roof band within
the library's range, one dominant component. Failures are kept in the index with a reason
(as `house_stats` does), never silently dropped. A `contact_sheet.py` sheet per 100
samples is the human review surface.

---

## 3. Model and training-data format

### 3.1 Task

Dense voxel-to-voxel translation at one fixed resolution: input channels from 2.5,
output two categorical maps (shape, material) over the same grid. Air is a shape class,
so occupancy is predicted too — which lets the model *sharpen* blobby TRELLIS geometry
into block-aligned walls and roofs, not only recolour it. Facing is part of the shape
class rather than a separate head so a stair can never be emitted without a direction.

### 3.2 Architecture (right-sized for one 16 GB GPU)

**3D residual U-Net**, 4 levels, base width 32 (32/64/128/256), GroupNorm, two 1×1×1
heads. About 6–8M parameters. Trained on random **64³ crops** in bf16 with batch 8:
~9–11 GB VRAM, ~0.7 s/iteration on the 4060 Ti; 100k iterations ≈ 20 hours. Inference
runs the full build with a sliding 64³ window (overlap 16, blended logits), so builds up
to the 128-block slider limit cost well under a second.

Why not the alternatives:

- **Sparse convolutions** (MinkowskiEngine, torchsparse) halve memory but are painful to
  build on Windows; at 64³ crops density is not the bottleneck.
- **Transformer over sparse voxels / TRELLIS-style SLat**: far more data-hungry than 15k
  samples justify; a U-Net's locality prior is the right bias for "stairs follow the slope".
- **Diffusion**: only worth it if the mapping is genuinely multi-modal (several valid
  roofs for one blob). Start deterministic; the eval set will show whether the
  predictions look averaged.

### 3.3 Training data format

```
data/synthetic/<dataset_version>/
  vocab.json                # materials[], shapes[], version — frozen per run
  index.csv                 # one row per sample: uid, sha256, licence, author, split,
                            #   target_size, seeds, qa_ok, qa_reason, domain (objaverse|lod2|real)
  samples/<uid>__<size>/
    target.schem            # canonical target, provenance in Metadata (2.5)
    input.npz               # occ, rgb, normal, origin, scale, yaw_deg
    target.npz              # shape (uint8), material (uint16)  — cache, rebuildable
    manifest.json
    preview.png
```

Kept **outside git** (5–10 GB) and outside `data/builds/` — the same rule `gen_houses`
enforces: generated data must never be mistaken for collected data. Axis order is Sponge
`(H, L, W)` everywhere so `schem_voxels`, `thumbs`, `contact_sheet` and `house_stats` read
it directly.

### 3.4 Training recipe

- **Loss**: cross-entropy on shape (class-weighted: air is ~78% of voxels at the library's
  median 22% density) + cross-entropy on material **masked to non-air target voxels** +
  a Dice term on occupancy. Material loss is also down-weighted on interior voxels (invisible
  in-game; `classify_shapes` already treats them as "cheapest to leave full").
- **Augmentation**: 90° yaw rotations and X/Z mirrors — **with the stair-facing
  permutation applied to the labels** (rotation maps E→S→W→N; a mirror swaps E↔W or N↔S);
  colour jitter on `rgb`; random crops biased to the surface.
- **Splits** by source *author*, not by object, so near-duplicate uploads by one artist do
  not leak; hold out 5% for validation and a fixed 500-sample test set.
- **Schedule**: AdamW, 3e-4 with cosine decay, 100k iterations, EMA weights; evaluate
  every 5k on synthetic-val and, once it exists, on the real set (section 4).
- **Metrics**: occupancy IoU, shape accuracy on surface voxels, material accuracy, and
  material *family* accuracy via `block_roles.family` (brick vs. bricks-of-another-shade is
  not an error a player sees); plus `house_stats` on 200 predicted schems as a
  distribution check (roof type mix, storey count, density).

### 3.5 Inference chaining (later, in `Real2Minecraft`)

`photo → _prepare_cutout → _run_trellis_api → normalize_mesh → estimate_yaw → voxel features
→ model → (shape, material) → apply_shapes-style state strings → export_schematic /
schem_write`. The model replaces `_quantize_colors + match_blocks + classify_shapes` (and
building mode's massing) behind a flag, with the current code as the fallback path.

---

## 4. Where the real, hand-curated schematics plug in

### 4.1 What has to be produced

Today: 106 real schems, 110 screenshots, 0 AI images, 0 meshes. Each real pair costs one
manual AI-photo generation (the step the app deliberately keeps manual) plus a TRELLIS run
(1–2 min GPU; `trellis_client` can drive it). Realistic budget: **50 builds** ≈ one long
day of human time. Pick them by `house_stats` coverage — every roof type, 1–3 storeys,
the main wall-material families — not by convenience.

### 4.2 Registration

A TRELLIS mesh of an AI photo of a render of a schem is not on the schem's grid. Align it
with deterministic code before it becomes an input: normalize + yaw-align, scale the
longest horizontal axis to the schem's known width/length, then a small search over
(yaw ∈ {0, 90, 180, 270} + ±5°, XZ shift ±3 blocks, scale ±10%) maximising occupancy IoU
between the voxelised mesh and the schem's `solid_mask()`. Record the residual IoU; pairs
below ~0.5 are TRELLIS failures, not training data.

### 4.3 Mapping the real schem into the vocabulary

- `block_roles.normalise` folds legacy names; each base block maps to the nearest
  `vocab.json` material by `block_colors.color_for` distance within the same
  `block_roles.family` (so `mossy_stone_bricks` → `stone_bricks`, not `moss_block`).
- Stairs and slabs keep shape + facing; other non-cubes become full cubes in v1.
- **Terrain and plants get an ignore label** (`block_roles.is_terrain/is_plant`): 55% of
  the library's non-air blocks are plot, the AI photo shows it, TRELLIS reconstructs it,
  and the model must not be graded on lawn. For the eval set, additionally crop the plot
  with the app's "Trim to the build" so the numbers describe the house.

### 4.4 Split and metrics

Split by **family**, not by build: `7__grannitlune`/`81__grannitlune-1` are byte
duplicates, `starfolk-town` ×4, `paris-house` ×4, `kollektive` ×3 and
`1-chunk-suburban-houses` ×10 are one source each. Roughly 30 builds for fine-tuning and
20 held out, the 16 synthetic `house_grammar` builds excluded from both. Metrics are the
ones in 3.4, computed on structural voxels only, plus the side-by-side schem/mesh
thumbnails the Library already renders on one camera — the same visual check the app uses
to catch a mis-dropped file.

### 4.5 Fine-tuning

Continue training from the synthetic checkpoint at 1/10 the learning rate for a few
thousand iterations on the 30 real pairs (heavy augmentation, mixed 1:4 with synthetic
batches so it does not forget). What this teaches that the synthetic target cannot:
Minecraft builders' conventions — stair courses on rakes, quoins and bands, window rhythm,
foundation courses — the things `house_grammar` had to encode by hand.

---

## 5. Decisions needed from you

1. **Pair asymmetry.** (a) synthetic degradation only, (b) synthetic + a 300–500 sample
   real-TRELLIS calibration subset (~10 GPU-hours, recommended), or (c) real TRELLIS for
   everything (~16 GPU-days). Without any of these the model is a lookup table of the
   voxeliser.
2. **Dataset and licence policy.** Objaverse-Sketchfab as primary (recommended). Then:
   CC-BY + CC0 only (safe for a commercial product, ~89% of objects), or also BY-SA / NC
   (more data, non-commercial or share-alike constraints on the derived dataset)? Include
   BuildingNet for eval (research-only terms)? Reserve textured LoD2 city models for a
   phase 2?
3. **Resolution envelope.** Target-size distribution sampled from the library (p10 16 /
   median 26 / p90 46 on the longest horizontal axis) and a 128-block maximum matching the
   `Real2Minecraft` slider; 64³ training crops. Also: plots excluded from synthetic targets
   and ignored in real ones (recommended), or modelled?
4. **Vocabulary scope for v1.** ~128 materials × {air, full, 2 slabs, 4 stairs}. Panes,
   fences, doors, trapdoors (13% of structural blocks in the library) deferred to v2?
5. **Where the code lives.** The voxel core (`voxelize`, `voxel_sampling`,
   `color_quantize`, `block_palette`, `block_shapes`, `mesh_normalize`, `yaw_align`) is in
   `Real2Minecraft`; the schem tooling (`schem_write`, `schem_voxels`, `block_roles`,
   `thumbs`, `contact_sheet`, `house_stats`) is here. Options: make `Real2Minecraft`'s core
   an installable package both repos depend on (recommended), or extract a third shared
   package. The generator script and training code then go in whichever repo owns the
   package.
6. **Compute and storage.** Local 4060 Ti only (sufficient: data generation ≈ 1.5 CPU-days,
   training ≈ 1 GPU-day per run), and where 100–165 GB of streamed downloads plus 5–10 GB
   of retained pairs go given 164 GB free on C: (external drive, or strict
   download-process-delete).
7. **Eval set size.** 50 real builds through the manual AI-photo + TRELLIS path
   (recommended) — that is the only remaining manual data work, and it gates every number
   in section 4.

## 6. Suggested build order (after the decisions)

1. Package boundary (Decision 5) and a `vocab.json` frozen from library statistics.
2. Ingest + filter: index from TRELLIS-500K/Objaverse metadata; a 200-object pilot
   downloaded, voxelised, palette-assigned, written as pairs, reviewed on a contact sheet.
3. Degradation model calibrated on the 228 cached TRELLIS outputs; the optional real-TRELLIS
   calibration subset.
4. Full generation run (streamed), QA report.
5. Model + training loop; first synthetic-only checkpoint and metrics.
6. Real pairs (section 4) produced through the app; registration tool; fine-tune; eval.
7. Inference flag in `Real2Minecraft`.

## 7. Unverified or open

- BuildingNet's licence text sits behind its access form; the paper names Trimble
  3D Warehouse as the source but states no terms.
- ArchiSet has no download page I could find; BuildingWorld's release is not confirmed.
- Berlin's building count and the exact set of PLATEAU municipalities with textured LoD2
  were not verified; both are secondary.
- TRELLIS-500K lists objects as of its curation date; some Sketchfab objects have since
  been removed, so the real yield will be a little lower than the caption counts.

## Sources

- Objaverse: <https://objaverse.allenai.org/>, metadata and LVIS files at
  <https://huggingface.co/datasets/allenai/objaverse>
- Objaverse-XL paper (licence statement, source counts): <https://arxiv.org/abs/2307.05663>
- TRELLIS-500K: <https://huggingface.co/datasets/JeffreyXiang/TRELLIS-500K>;
  toolkit: <https://github.com/microsoft/TRELLIS/blob/main/DATASET.md>;
  glb export axes: <https://github.com/microsoft/TRELLIS/blob/main/trellis/utils/postprocessing_utils.py>
- Cap3D captions: <https://huggingface.co/datasets/tiange/Cap3D>
- Objaverse++: <https://github.com/TCXX/ObjaversePlusPlus>
- BuildingNet: <https://github.com/buildingnet/buildingnet_dataset>, paper <https://arxiv.org/abs/2110.04955>
- 3D Warehouse terms FAQ: <https://help.sketchup.com/en/3d-warehouse/3d-warehouse-terms-use-faq>
- 3D BAG: <https://docs.3dbag.nl/en/>
- PLATEAU (CC BY 4.0, textured subset): <https://collab.dvb.bayern/spaces/TUMgisproject/pages/64282282/Semantic+3D+City+Model+of+Tokyo>;
  open city model catalogue: <https://github.com/OloOcki/awesome-citygml>
- Berlin LoD2: <https://daten.berlin.de/datensaetze/lod2-geb-udedaten-berlin>,
  <https://www.businesslocationcenter.de/downloadportal>
- Helsinki 3D: <https://www.hel.fi/en/decision-making/information-on-helsinki/maps-and-geospatial-data/helsinki-3d>
- Building3D: <https://building3d.ucalgary.ca/>
- BuildingWorld: <https://arxiv.org/abs/2511.06337>
- ArchiSet: <https://mlanthology.org/iccv/2025/yin2025iccv-archiset/>
- 3D-FRONT: <https://arxiv.org/abs/2011.09127>; ShapeNet: <https://shapenet.org/about>
