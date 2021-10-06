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
    orcid: 0000-0003-1456-7967
    affiliation: 1
  - name: Florian Wiesinger
    orcid: 0000-0002-5597-6057
    affiliation: "1,2"
affiliations:
  - name: Department of Neuroimaging, King's College London
    index: 1
  - name: GE Healthcare
    index: 2
date: 2021-06-25
bibliography: paper.bib
---

# Summary

- Standard MRI methods acquire Fourier-encoded data on a regularly spaced Cartesian grid. Noncartesian MRI methods abandon this grid and instead acquire data along arbitrary trajectories in k-space, which can lead to advantages such as motion robustness and more flexible pulse sequence design.
- Zero Echo Time (ZTE) imaging is a specialised form of MRI with an inherently noncartesian three-dimensional radial trajectory [@Ljungberg]. ZTE imaging has many interesting benefits, such as near silent operation, but also technical challenges such as dead-time gap artefacts [@Froidevaux].
- We have developed a toolbox, named Radial Interstices Enable Speedy Low-volume imagING (RIESLING), tuned for high performance reconstruction of MRI data acquired with 3D noncartesian k-space trajectories. While our group has a focus on ZTE imaging, RIESLING is suitable for all 3D noncartesian trajectories.

# Statement of Need

3D noncartesian trajectories can be challenging to reconstruct compared to other MRI trajectories. The major issues are:

- **Inseparability of the trajectory dimensions.** In Cartesian imaging it is often possible to consider one of the image dimensions separately to the others, leading to algorithmic complexity reduction and memory savings. 3D noncartesian must consider all dimensions simultaneously.
- **Oversampling requirements.** In order to obtain good image quality, it is necessary to oversample the reconstruction grid. Coupled with the inseparability of the 3D problem, this leads to large memory requirements.
- **Sample density compensation.** Noncartesian trajectories can lead to uneven k-space sampling, the correction of which is equivalent to preconditioning a linear system [@cgSENSE]. For 2D noncartesian trajectories, it is generally not necessary to correct for this in the context of an iterative reconstruction, as the problem will still converge. For 3D noncartesian trajectories, such preconditioning is essential to ensure reasonable convergence properties.

Existing MRI reconstruction toolboxes such as BART [@BART] or SigPy [@SigPy] provide high-quality implementations of many reconstruction algorithms but have prioritised applications other than 3D noncartesian reconstruction. We hence created a dedicated toolbox for this, including specific features such as:

- A configurable grid oversampling factor
- A thread-safe gridding implementation suitable for multi-core CPUs
- Integrated sample density compensation functions [@Zwart]

RIESLING (http://github.com/spinicist/riesling) is written with a modern C++ toolchain and utilizes Eigen [@Eigen] for all core operations, providing high performance. Data is stored in the HDF5 format allowing easy integration with other tools. We provide multiple reconstruction strategies, including classic conjugate-gradient SENSE [@cgSENSE] and Total-Generalized Variation [@TGV]. We are actively using RIESLING for our own studies and hope that it will be of interest to other groups using such sequences.

# References
