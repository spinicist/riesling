.. RIESLING documentation master file, created by
   sphinx-quickstart on Sun May  9 11:26:39 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

RIESLING
========

.. image:: ../riesling-logo.png
   :alt: RIESLING Logo

Radial Interstices Enable Speedy Low-volume imagING (RIESLING) is a tool for reconstructing non-cartesian MRI scans. It was developed for reconstructing 3D radial center-out trajectories associated with Zero Echo-Time (ZTE) or Ultrashort Echo-Time (UTE) sequences. These trajectories provide unique challenges in efficient reconstruction compared to Cartesian trajectories (both 2D and 3D). However it is capable of reconstructing any trajectory.

.. toctree::
   :maxdepth: 1
   :caption: Contents

   faq
   data
   recon
   util
   op

Commands
--------

RIESLING is provided as a single executable file. The ``riesling`` executable provides multiple individual commands, which are grouped into reconstruction, data manipulation, sensitivity map estimation, basis creation, linear operators, and utilities. To see a full list of commands currently available, run ``riesling`` with no arguments. Detailed help for commands can be found in the category pages: :doc:`recon`, :doc:`op` and :doc:`util`. The most useful are:

- ``riesling recon-lsq``  Least-squares reconstruction including sensitivity maps.
- ``riesling recon-rlsq`` Regularized least-squares reconstruction, similar to ``bart pics``.
- ``riesling denoise``    Denoise an already reconstructed image.
- ``riesling h5`` Prints information about compatible ``.h5`` files

RIESLING exploits the inherent oversampling of most non-Cartesian trajectories at the center of k-space to generate SENSE maps directly from the input data, but utilities are provided to explicitly extract sensitivities if desired. Further details can be found in :doc:`util`.

Examples
--------

A `tutorial notebook <https://github.com/spinicist/riesling-examples/tutorial.ipynb>`_ is provided to explain the basic steps in reconstruction.

Input Data
----------

An important step with using RIESLING is providing data in the correct ``.h5`` format. Details of this format can be found in :doc:`data`. Matlab and Python code to generate these files is provided in the ``/matlab`` and ``/python`` directories of the repository respectively. Users of the ZTE sequence on GE platforms should contact the authors to discuss conversion strategies.
