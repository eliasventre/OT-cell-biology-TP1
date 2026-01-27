# OT-cell-biology-TP1
Practical session 1 — Entropic Optimal Transport for trajectory inference in single-cell biology

This repository contains the material for the first practical session
of the course *Optimal Transport & Machine Learning* (M2 Applied Mathematics).

## Topic
Trajectory inference from single-cell data using **entropic optimal transport**.

## Contents
- `notebooks/`: guided Jupyter notebooks for the TP
- `data/`: simulated datasets
- `src/`: helper functions

## Requirements
Python ≥ 3.9  
Main libraries: numpy, scipy, matplotlib, POT, pytorch

## Installation

### Option 1 — Google Colab (recommended)

If you encounter installation issues locally, you can run the TP on Google Colab.

```python
!git clone https://github.com/eliasventre/OT-cell-biology-TP1.git
%cd OT-cell-biology-TP1
!pip install -r requirements.txt```


### Option 2 — Conda

conda env create -f environment.yml
conda activate ot-cell-biology
jupyter notebook

### Option 3

pip install -r requirements.txt
