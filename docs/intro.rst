Introduction
============

RIESLING is a tool for reconstructing non-cartesian MRI scans. It has been tuned specifically for reconstructing 3D radial center-out trajectories associated with Zero Echo-Time (ZTE) or Ultrashort Echo-Time (UTE) sequences. These trajectories provide unique challenges in efficient reconstruction compared to Cartesian trajectories (both 2D and 3D).

RIESLING is provided as a single executable file, similar to ``bart``. The ``riesling`` executable provides multiple individual commands, which vary from basic utilities to interrogate files to complete reconstruction pipelines. To see a full list of commands currently available, run ``riesling``. The full list is not repeated here as they are subject to change. However, the core commands are:

- ``riesling hdr`` Prints the header information from compatible ``.h5`` files
- ``riesling recon`` Performs a non-iterative reconstruction with either root-sum-squares (``--rss``) channel combination or a sensitivity-based combination (the default)
- ``riesling sense`` Extract channel sensitivities from a dataset for future use
- ``riesling sdc`` Pre-calculate sample densities
- ``riesling plan`` Pre-plan FFTs
- ``riesling cg`` Iterative conjugate-gradients SENSE reconstruction
- ``riesling admm`` Regularized cgSENSE reconstruction (only Locally-Low-Rank available currently)
- ``riesling tgv`` TGV-regularized iterative reconstruction

A `tutorial notebook <https://github.com/spinicist/riesling-examples/tutorial.ipynb>`_ can be run interactively at on `MyBinder <https://mybinder.org/v2/gh/spinicist/riesling-examples/HEAD?filepath=tutorial.ipynb>`_. This explains the various steps required to generate a simulated phantom dataset and then reconstruct it. You will need to reduce the matrix size to 64 to run with MyBinder's RAM limit.

An important step with using RIESLING is providing data in the correct ``.h5`` format. Details of this format can be found in :doc:`data`. Users of the ZTE sequence on GE platforms should contact the authors to discuss conversion strategies.

Further details about the reconstruction tools can be found in :doc:`recon`.
