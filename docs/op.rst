Operators
=========

The MRI reconstruction problem can best be represented by a series of linear operators. In the full reconstruction commands (:doc:`recon`) these are already combined into pipelines to complete the full transform from non-cartesian data to final image. However, sometimes it is useful to have access to the key steps and hence the operators are also exposed as commands.

Note that all of these operators are defined with the forward, default, direction going from image space towards non-cartesian space. Hence you likely want to specify the ``--adj`` option.

* `grid`_
* `nufft`_
* `pad`_
* `reg`_
* `sense`_

grid
----

``riesling grid`` will carry out only the first step of the NUFFT, i.e. it will grid non-Cartesian k-space to Cartesian (or vice versa) and save the result. This can be useful to check that a dataset has been acquired correctly.

To diagnose trajectory and sample density issues, you can instead use ``riesling traj``.

*Usage*

.. code-block:: bash

    riesling grid file.h5 --adj

*Input/Output*

For the forward operation, ``file.h5`` must contain the dataset ``cartesian``, and the output ``file-grid.h5`` will contain the ``noncartesian`` dataset. The adjoint operation must contain ``noncartesian`` in the input and will produce ``cartesian`` in the output.

*Important Options*

* ``--adj``

    Apply the adjoint operation, i.e. from non-cartesian to cartesian.

* ``--sdc=none,pipe,pipenn,file.h5``

    Choose the sample density compensation scheme. The default is ``pipenn``. ``none`` means no density compensation. ``pipe`` is the full Pipe/Zwart/Menon density compensation. ``pipenn`` is a fast approximation to ``pipe``. ``file.h5`` uses a pre-computed sample density from the specified file.

nufft
-----

Applies a NUFFT - i.e. combines gridding and an FFT.

*Usage*

.. code-block:: bash

    riesling nufft file.h5 --adj

*Input/Output*

The forward operation expects a dataset ``channels`` and outputs a dataset ``noncartesian``. The adjoint operation expects a dataset ``noncartesian`` and outputs a dataset ``channels``.

*Important Options*

* ``--adj``

    Apply the adjoint operation.

* ``--traj=file.h5``

    Use the trajectory from a different file for gridding.

pad
---

The forward operation pads an image, the adjoint crops an image.

*Usage*

.. code-block:: bash

    riesling pad file.h5 X,Y,Z

The second option ``X,Y,Z`` is a comma-delimited set of numbers indicating the required output dimensions. The other dimensions of the ``image`` dataset (frames and volumes) are not affected.

*Output*

``file-pad.h5`` containing the padded/cropped dataset.

*Important Options*

* ``--adj``

    Apply the adjoint operation (cropping).

* ``--channels``

    Changes the operation to work on 6D ``channels`` datasets. The extra dimension (channels) is also not affected.

reg
---

Applies regularization to an image. Useful to check what the impact of the regularizer during ``admm`` will be.

*Usage*

.. code-block:: bash

    riesling reg file.h5 --llr --patch-size=N --lambda=L

*Output*

``file-reg.h5`` containing the regularized ``image`` dataset.

*Important Options*

* ``--llr``

    Use Locally-Low-Rank regularization.

* ``--lambda=F``

    The regularization parameter.

* ``--patch-size=N``

    The patch-size to apply local regularizers on.

* ``--slr``

    Apply Structured Low-Rank regularization. Acts on a ``channels`` dataset, not ``image``.

sense
-----

Applies SENSE channel combination (adjoint operation) or splitting (forward operation).

*Useage*

.. code-block:: bash

    riesling sense file.h5 sense.h5 --adj

*Input/Output*

Forward operation requires ``file.h5`` containing a dataset ``channels``, outputs ``file-sense.h5`` containing ``image``. Adjoint operation requires the ``image`` dataset, outputs ``channels``.

The SENSE maps contained in ``sense.h5`` must match the spatial dimensions of the dataset in ``file.h5``.

*Important Options*

* ``-adj``

    Apply the adjoint operation (SENSE channel combination)
