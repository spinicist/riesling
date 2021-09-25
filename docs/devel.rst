Notes for Developers
====================

These notes are intended to highlight to give a brief overview of the RIESLING code. They are not intended as a comprehensive API reference, but may evolve into that over time.

Unit Tests
----------

RIESLING includes some unit tests which cover the core functionality. They do not cover everything, and should likely be expanded. The tests use the `Catch2 <https://github.com/catchorg/Catch2>`_ framework.

The tests are built with the ``riesling-tests`` target, which is included in the ``all`` target. ``riesling-tests`` does not have an install location set, so it will be built but remain in the build directory if the ``install`` target is used.

To run the tests, simply run ``./riesling-tests`` within the build directory. By default, all cases are tested. A list of individual test cases can be obtained with ``riesling-tests --list-tests``.

The test case code is located in ``/test``. Generally the files are named after the corresponding file in ``/src``, unless it makes sense to sub-divide the tests further.

Code Structure
--------------

RIESLING is built as a monolithic binary. There is a simple ``main.cpp`` file which specifies the available commands, each of which is contained in a ``main_command.cpp`` file. The commands can roughly be split into simple utilities (such as ``riesling hdr``) and the more complicated commands that perform a particular reconstruction method (such as ``riesling cg``).

The most important command to explain is ``riesling cg`` / ``main_cg.cpp``. The top of the ``main_cg`` function is fairly straight-forward - a set of flags is declared, including those common across recon methods which are defined by a macro defined in ``parse_args.h``. Once the command-line has been parsed, the trajectory and ``Info`` header struct are read from the input file.

The next few lines initialise the gridding kernel and the ``GridOp`` object. The ``Trajectory`` and ``GridOp`` objects are the work-horses of RIESLING. ``Trajectory`` can calculate an efficient ``Mapping`` between non-Cartesian and Cartesian co-ordinates. This consists of lists of matching integer non-Cartesian and Cartesian co-ordinates, the floating-point offset from the Cartesian grid-point, and a list of indices sorted by Cartesian grid location. This ``Mapping`` depends on the chosen over-sampling factor. This enables fast, thread-safe, non-Cartesian to Cartesian gridding as each thread can work on a section of the Cartesian grid without conflicting writes. The ``Mapping`` is used to construct the ``GridOp`` object, which contains the interpolation code.

After the ``GridOp`` is constructed, the Sample Density Compensation is either calculated or loaded from a file on disk. Then the necessary cropping between the oversampled reconstruction grid and the output image is calculated, followed by the apodization required to correct for any apodization introduced by the gridding kernel. Next the required FFT for the reconstruction grid is planned using the FFTW library.

The next key step is calculating the sensitivity maps, or loading them from disk. Calculating them currently requires constructing a second ``GridOp`` object within the ``DirectSENSE`` function. This uses a very low resolution, so constructing and sorting the grid co-ordinates is usually fast.

At this point we have all the basic building blocks of a reconstruction pipeline, so we proceed to set one up. Generally this means constructing operators that take us from non-Cartesian data to image and vice versa. C++ lambda functions are used to provide functions that act as the overall Linear Operator/System. These functions are then passed to the iterative optimization routines. For conjugate gradients we require two operators - a decoding operator that goes from non-Cartesian data to image, and a system operator that applies both encoding and decoding (i.e. maps from the image to the image, passing through the system).

After these lambda functions have been constructed, we loop through all the volumes in the input file, load the data, apply the decoding operator to get a starting image, and then apply conjugate gradients to optimize the image. We crop to the desired output FoV and apply a Tukey filter if requested by the user. Finally the images are written to disk.
