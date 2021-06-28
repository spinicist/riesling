---
title: "Radial Interstices Enable Speedy Low-volume Imaging"
tags:
  - mri
  - reconstruction
  - ZTE
  - cpp
authors:
  - name: Tobias C Wood
    orcid: 0000-0001-7640-5520
    affiliation: 1
  - name: Emil Ljungberg
    orcid:
    affiliation: 1
  - name: Florian Wiesinger
    affiliation: 1,2
affiliations:
  - name: Department of Neuroimaging, King's College London
    index: 1
  - name: GE Healthcare
    index: 2
date: 2021-06-25
bibliography: paper.bib
---

# Summary

- 3D non-cartesian MRI sequences present unique reconstruction challenges including large memory requirements and convergence problems due to density compensation [@Zwart]. Existing reconstruction toolboxes such as BART [@BART] or SigPy [@SigPy] provide high-quality implementations of many algorithms but have not been tuned for the specific case of 3D non-cartesian, which can lead to suboptimal performance.
- The RIESLING toolbox (http://github.com/spinicist/riesling) has been written and tuned for the specific application of 3D radial ZTE imaging, but can be used to reconstruct other non-cartesian trajectories as well. RIESLING is written with a modern C++ toolchain and utilizes Eigen [@Eigen] for all core operations, providing high performance. Data is stored in the HDF5 format allowing easy interoperation with other tools.
- RIESLING provides multiple reconstruction strategies, including classic conjugate-gradient SENSE [@cgSENSE] and Total-Generalized Variation [@TGV].
- We are actively using RIESLING for our own studies and hope that it will be of interest to other groups using such sequences.

# References