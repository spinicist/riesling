Introduction
============

RIESLING is a tool for reconstructing non-cartesian MRI scans. It has been developed specifically for reconstructing 3D radial center-out trajectories associated with Zero Echo-Time (ZTE) or Ultrashort Echo-Time (UTE) sequences. These trajectories provide unique challenges in efficient reconstruction compared to Cartesian trajectories (both 2D and 3D). However it is capable of reconstructing any non-Cartesian trajectory.

Commands
--------

RIESLING is provided as a single executable file, similar to ``bart``. The ``riesling`` executable provides multiple individual commands, which vary from basic utilities to interrogate files to complete reconstruction pipelines. To see a full list of commands currently available, run ``riesling``. Detailed help for each command can be found in the category pages: :doc:`util`, :doc:`op`, and :doc:`recon`. The full list is not repeated here as they are subject to change. However, the most useful are:

- ``riesling hdr`` Prints the header information from compatible ``.h5`` files
- ``riesling recon`` Performs a non-iterative reconstruction with either root-sum-squares (``--rss``) channel combination or a sensitivity-based combination (the default)
- ``riesling lsqr`` Iterative SENSE reconstruction

RIESLING exploits the inherent oversampling of most non-Cartesian trajectories at the center of k-space to generate SENSE maps directly from the input data, but utilities are provided to explicitly extract sensitivities if desired. Further details can be found in :doc:`util`. Internally, similarly to BART, RIESLING pipelines are constructed as a series of Linear Operators. These operators are exposed as individual commands if you wish to build up your own pipele, see :doc:`op`.

Examples
--------

A `tutorial notebook <https://github.com/spinicist/riesling-examples/tutorial.ipynb>`_ can be run interactively at on `MyBinder <https://mybinder.org/v2/gh/spinicist/riesling-examples/HEAD?filepath=tutorial.ipynb>`_. This explains the various steps required to generate a simulated phantom dataset and then reconstruct it.

Input Data
----------

An important step with using RIESLING is providing data in the correct ``.h5`` format. Details of this format can be found in :doc:`data`. Matlab code to generate these files is provided in the ``/matlab`` directory of the repository. Users of the ZTE sequence on GE platforms should contact the authors to discuss conversion strategies.
