MLX C
=====
Version |version|.

MLX C is a C API for `MLX <https://github.com/ml-explore/mlx>`_.

MLX is an array framework for machine learning on Apple silicon. MLX C expands
MLX to the C language, making research and experimentation easier on Apple
silicon.

MLX C can be used standalone or as a bridge to bind other languages
to MLX. For example, the `MLX Swift
<https://github.com/ml-explore/mlx-swift/>`_ package uses MLX C for Swift
bindings to MLX.

.. toctree::
   :caption: Installation
   :maxdepth: 1

   install

.. toctree::
   :caption: Overview
   :maxdepth: 1

   overview

.. toctree::
   :caption: Object Reference
   :maxdepth: 1

   array
   device
   stream
   string
   vector
   map
   optional
   closure
   distributed_group

.. toctree::
   :caption: API Reference
   :maxdepth: 1

   ops
   fft
   linalg
   random
   io
   transforms
   distributed_ops
   compile
   fast
   metal
   export
   error
