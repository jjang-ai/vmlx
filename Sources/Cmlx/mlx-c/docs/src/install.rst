Building and Installing
=======================

CMake is required to build MLX C. You can install it with `Homebrew <https://brew.sh/>`_:

.. code-block:: shell

  brew install cmake

To build MLX C, run the following commands:

.. code-block:: shell

  cmake -B build -DCMAKE_BUILD_TYPE=Release
  cmake --build build -j

MLX C will fetch `MLX <https://github.com/ml-explore/mlx>`_ under the hood,
compile it, and then compile the C API.
