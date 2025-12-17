# DAQplugin

DAQplugin is a collection of tools for computing, visualizing, and exporting **DAQ scores** for protein atomic models in cryo-EM maps.

This repository provides:

- Google Colab ready Jupyter notebooks for DAQ score computation and NPY file generation  
- A ChimeraX plugin (`daqcolor`) for interactive coloring and visualization  
- Command-line utilities for processing and file export  

DAQ and DiffModeler are included as Git submodules to ensure consistency with published methods.

---

## Repository Structure

```
DAQplugin/
├── DAQ/                  # DAQ core (git submodule)
├── DiffModeler/          # DiffModeler core (git submodule)
├── daqcolor/             # ChimeraX plugin
│   ├── src/
│   ├── bundle_info.xml
│   └── 00README.txt
├── cli/                  # Command-line scripts
├── map_util/             # Map preprocessing utilities (zarr v3)
├── DAQ_Score.ipynb       # DAQ score calculation notebook
├── DAQ_Score_Grid.ipynb  # Grid / NPY generation notebook
├── README.md
└── LICENSE
```

---

## Installation

### Clone the Repository (IMPORTANT)

This repository uses **Git submodules**.

Clone with submodules enabled:

```bash
git clone --recurse-submodules https://github.com/gterashi/DAQplugin.git
```

If you already cloned without submodules:

```bash
git submodule update --init --recursive
```

---

## 1. DAQ Score Computation (Jupyter Notebook on Google Colab)

### Notebook

- `DAQ_Score_Grid.ipynb`

### Purpose

This notebook computes:

- DAQ scores from atomic models (PDB/CIF) and cryo-EM maps (MRC/MAP)
- Numpy files (`.npy`) containing per-point probability and score information

The generated `.npy` files are used by the ChimeraX plugin (`daqcolor`) for visualization.

### Typical Workflow

1. Provide:
   - Atomic model (`.pdb` or `.cif`)
   - Cryo-EM map (`.mrc` or `.map`)
2. Run the notebook cells sequentially
3. Output:
   - `points_AA_ATOM_SS_swap.npy`
   - Optional: PDB file with DAQ score

---

## 2. ChimeraX Plugin: `daqcolor`

The `daqcolor` plugin enables **interactive coloring and visualization of DAQ scores** in ChimeraX.

### Installation (Developer Mode)

From the ChimeraX command line:

```bash
# Uninstall (if already installed)
devel clean [DAQplugin PATH]/daqcolor

# Install
devel install [DAQplugin PATH]/daqcolor
```

> **Note**  
> The `devel` command requires ChimeraX developer tools.

---

### Help

```bash
help daqcolor
```

---

### Commands

#### Apply DAQ coloring once

```
daqcolor apply npyPath model [halfwindow k] [colormap] [metric] [atomName] [clampMin] [clampMax]
```

- `npyPath` : Path to the numpy file computed by NoteBook.  
- `model`   : ChimeraX model ID (e.g., `#1`)  
- `halfwindow k`       : Window averaging parameter def:9  
- `metric`  :
  - `aa_score` — DAQ(AA) score  
  - `atom_score` — DAQ(CA) score  
  - `aa_conf:<AA>` — DAQ confidence for a specific amino-acid type  
- `atomName` : Atom name (default: CA)  
- `clampMin`, `clampMax` : Optional score clamping  

**Examples**

```bash
# Color model #2 by amino-acid DAQ score
daqcolor apply ./points_AA_ATOM_SS_swap.npy #2 metric aa_score 

# Color by atom (CA) DAQ score
daqcolor apply ./points_AA_ATOM_SS_swap.npy #1 metric atom_score k 1
```

---

#### Live recoloring

```
daqcolor monitor model [npyPath] [k] [colormap] [metric] [atomName] [on true|false]
```

**Example**

```bash
daqcolor monitor #2 ./points_AA_ATOM_SS_swap.npy metric aa_score on true
```

Stop monitoring:

```bash
daqcolor monitor #2 on false
```

---

#### Visualize point clouds

```
daqcolor points npyPath [radius] [metric] [colormap] [clampMin] [clampMax]
```

**Example**

```bash
daqcolor points ./points_AA_ATOM_SS_swap.npy radius 0.6 metric aa_score
```

### Clear markers:

```bash
daqcolor clear
```

---

### Saving Colored Models

Once colored, models can be exported using ChimeraX:

Save #1 as colored.pdb
```bash
save colored.pdb #1
```

- DAQ scores are written to the **B-factor field**
- Window-averaged scores (defined by `halfwindow k`) are preserved
- Both PDB and CIF formats are supported

---

## 3. Command-Line Usage (CLI)
### DAQ Score Export to B-factor (CLI)

The script **daq_write_bfactor.py** writes DAQ-style scores into the B-factor field of a protein structure file (PDB or mmCIF), using the same scoring logic as the ChimeraX daqcolor plugin.

### Requirements
- Python 3.8+
- NumPy
- SciPy (optional, for fast kNN; NumPy fallback is used if unavailable)
- gemmi (required for PDB/mmCIF I/O)

### Install dependencies:
```
pip install numpy scipy gemmi
```

### Basic Usage
```
python daq_write_bfactor.py \
    -i model.cif \
    -p points_AA_ATOM_SS_swap.npy \
    -m aa_score \
    -o model.daq.b.cif
```

This command:

- Computes DAQ scores per residue
- Writes the scores to the B-factor field
- Preserves the input file format (PDB or mmCIF)

### Command-Line Options
```
-i, --input        Input structure file (.pdb/.cif/.mmcif) [required]
-o, --output       Output structure file (.pdb/.cif/.mmcif) [required]
-p, --points       Points file (N×32 numpy file) [required]

-m, --metric       Scoring metric:
                     aa_score        DAQ(AA) score (per-residue)
                     atom_score      DAQ(CA) score
                     aa_conf:ALA     Confidence for a specific AA type

--atom-name        Atom name used to define residue coordinates (default: CA)
-k                 Number of nearest neighbors for kNN (default: 1)
--radius           Distance cutoff for kNN in Å (default: 3.0; <=0 disables)
--half-window      Window averaging half-width (n±half_window, default: 9)
--no-window        Disable window averaging
--nan-fill         Value written when score is NaN/inf (default: 0.0)
```

### Scoring Metrics
- aa_score	DAQ score for the native residue type
- atom_score	DAQ score based on CA atom probability
- aa_conf:XXX	DAQ confidence for a specific amino acid (e.g. aa_conf:ALA)

### Window Averaging
By default, scores are smoothed using chain-aware window averaging:

Residues within
- residue_number ± half_window
(default: ±9 residues) are averaged
- Only residues in the same chain are considered
- Non-finite values are ignored

### Disable window averaging:
```
--no-window
```

---
## Notes

- DAQ and DiffModeler are included as submodules to ensure consistency.
- The ChimeraX plugin is intended for visualization and inspection.
- Numerical analysis should be performed via notebooks or CLI tools.
- This repository is under active development.

---

## License

See the `LICENSE` file for details.
