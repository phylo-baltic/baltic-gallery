---
date: 2026-09-22
---

# baltic v1.0.0 (Cedar) released

`baltic` 1.0.0 is a major, intentionally backwards-incompatible release. Compared with the previous PyPI release, 0.3.0, it reorganizes the package into focused modules, modernizes the public API, and adds a broad set of tree-manipulation, analysis, and visualization tools.

### Highlights

- Reorganized the former monolithic implementation into dedicated modules for trees, branch types, I/O, utilities, advanced visualizations, and posterior-tree processing.
- Added an explicit tree model: `Tree` and parsing functions now require `treeType="divergence"` or `treeType="time"`.
- Standardized most public classes and methods on `PascalCase` and `snake_case` names.
- Added rerooting, midpoint rooting, continuous root-to-tip regression rooting, tree rescaling, tree reduction, tree condensation, and trait-based tree explosion.
- Unified rectangular, circular, and unrooted rendering behind `plot_tree()`, `plot_points()`, and `plot_text()`.
- Added advanced visualisation support for tanglegrams, tangled tree chains, Muller plots, skygrids, root-to-tip regressions, maps, SNP alignments, reticulations, uncertainty intervals, gradient clades, and tree-plus-matrix plots.
- Added import and export support for Auspice v2 JSON, including Nextstrain-hosted datasets.
- Rebuilt and substantially expanded the documentation, API reference, examples, and plotting tutorials.

### Breaking changes and migration

The 0.3.0 API is not backwards-compatible with 1.0.0. Common migrations include:

| 0.3.0 | 1.0.0 |
| --- | --- |
| `bt.tree`, `bt.node`, `bt.leaf` | `bt.Tree`, `bt.Node`, `bt.Leaf` |
| `bt.clade`, `bt.reticulation` | `bt.Clade`, `bt.Reticulation` |
| `bt.make_tree(data)` | `bt.make_tree(data, treeType="divergence")` |
| `bt.make_treeJSON(...)` | `bt.make_tree_JSON(..., treeType=...)` |
| `bt.loadNewick(...)` | `bt.io.load_newick(..., treeType=...)` |
| `bt.loadNexus(...)` | `bt.io.load_nexus(..., treeType=...)` |
| `bt.loadJSON(...)` | `bt.io.load_JSON(..., treeType=...)` |
| `tree.getExternal()`, `getInternal()`, `getBranches()` | `tree.get_external()`, `get_internal()`, `get_branches()` |
| `tree.renameTips()`, `sortBranches()` | `tree.rename_tips()`, `sort_branches()` |
| `tree.commonAncestor(...)` | `tree.find_MRCA(...)` |
| `tree.setAbsoluteTime(...)` | `tree.set_absolute_time(...)` |
| `tree.toString(...)` | `tree.to_string(...)` |
| `tree.reduceTree(...)` | `tree.reduce_tree(...)` |
| `tree.collapseSubtree(...)` | `tree.collapse_subtree_to_clade(...)` |
| `tree.uncollapseSubtree()` | `tree.restore_all_collapsed_subtrees()` |
| `tree.plotTree()` / `plotCircularTree()` | `tree.plot_tree(..., treeType="rectangular" | "circular" | "unrooted")` |
| `tree.plotPoints()` / `plotCircularPoints()` | `tree.plot_points(..., treeType=...)` |
| `tree.addText()` / `addTextCircular()` / `addTextUnrooted()` | `tree.plot_text(..., treeType=...)` |

Loader functions remain re-exported at package level, but new code should use the explicit `bt.io` namespace shown above.

Minimal 1.0.0 usage:

```python
import baltic as bt

tree = bt.make_tree(
    "((A:1.0,B:2.0):1.0,C:3.0);",
    treeType="divergence",
)

nexus_tree = bt.io.load_nexus(
    "example.tree",
    treeType="time",
)
```

### Added

#### Tree model and manipulation

- Introduced `BranchLike` as the shared base class for `Node`, `Leaf`, `Clade`, and `Reticulation`.
- Added branch path and sibling helpers with `get_path_to_root()` and `get_siblings()`.
- Added `get_leaf()` for exact single-tip lookup and expanded `find_MRCA()` to accept tip names or branch objects as separate arguments.
- Added `reroot()`, `midpoint_root()`, and `root_by_regression()`. The new regression implementation uses a continuous closed-form edge search, supports uncertain tip dates, and retains the earlier search as `root_by_regression_legacy()`.
- Added `rescale()`, `reduce_tree()`, `condense_tree()`, `explode_tree()`, and `state_collapse_tree()`.
- Added dictionary-based tree statistics through `treeStatsDict()`.
- Added export to Auspice v2 JSON with `to_auspice_json()`.

#### Plotting and analysis

- Added unified rectangular, circular, and unrooted layouts for trees, points, and labels.
- Added aligned tip labels, exploded-tree plotting, collapsed-clade rendering, scale bars, and calendar or numeric deep-time grids.
- Added node probability bars, treemaps, pie charts, TMRCA posterior plots, and highest-posterior-density utilities.
- Added tanglegrams and tangled chains for comparing tree topologies.
- Added root-to-tip, skygrid, Muller, map-connection, reassortment/reticulation, SNP-alignment, gradient-clade, and tree-matrix visualizations.
- Added colour-map construction and desaturation helpers, Bezier helpers, path effects, and scientific-notation formatting.

#### Input, output, and posterior processing

- Added a dedicated `baltic.io` module with Newick, Nexus, and Auspice JSON loaders plus explicit tip-date processing.
- Added support for partially specified sampling dates, BEAST 2 Boolean annotations, and travel-aware phylogeographic trees.
- Added posterior-tree iteration and multiprocessing helpers in `baltic.samogitia` for lineage-state tracing, TMRCA extraction, and tree-length processing.
- Added logging throughout the package in place of most ad hoc diagnostic output.

### Changed

- Tree branch-length semantics are now explicit through `treeType`; only `"divergence"` and `"time"` are accepted.
- Public loaders and most tree operations now use PEP 8-style names and more consistent callback arguments.
- Topology-only trees without branch lengths are handled by assigning ultrametric branch lengths with a total height of 1.0.
- Tree parsing and annotation handling were made more robust for Newick, Nexus, BEAST, and Auspice inputs.
- Circular, unrooted, and rectangular plotting now share a common API and styling model based on fixed values or callback functions.
- Runtime dependencies now include `scipy` and `requests`, in addition to `numpy` and `matplotlib`.

### Fixed

- Corrected circular and unrooted coordinate generation and plotting behavior.
- Fixed subtree extraction when `stem=False` and strengthened `reduce_tree()` validation.
- Fixed Auspice JSON import behavior and added reliable round-trip export support.
- Fixed tree-to-map connections, branch sorting and auto-sorting interactions, TMRCA and SNP-alignment plotting, uncertain-date assignment, and several `samogitia` workflows.
- Improved handling of trees with missing branch lengths, Nexus tip names, rerooting edge cases, and root-to-tip regression.

### Documentation

- [Read the Docs](https://baltic.readthedocs.io/)
- [Gallery of tutorials and examples](https://phylo-baltic.github.io/baltic-gallery/)