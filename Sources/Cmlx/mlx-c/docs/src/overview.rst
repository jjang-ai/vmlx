Overview
========

MLX C wraps and follows as closely as possible the C++ API of `MLX
<https://github.com/ml-explore/mlx>`_.

C Objects
---------

MLX C relies on several opaque C ``struct`` to operate. This includes:

* :doc:`Arrays <array>`, on which computations are performed: :class:`mlx_array`.
* :doc:`Devices <device>`, which define the compute unit where operations are performed: :class:`mlx_device`.
* :doc:`Streams <stream>`, which ingest and execute array operations on a specific device: :class:`mlx_stream`.

Other ``struct`` objects holding data, but not directly related to compute,
are also available, such as :doc:`strings <string>`, :doc:`vectors
<vector>` and :doc:`maps <map>`. There are also few extra objects such as
:doc:`closures <closure>` (which encapsulate specific function signatures,
possibly holding upvalues), and :doc:`distributed groups
<distributed_group>` (which enable distributed computation).

All these objects have in common the way they operate: one need to create
them with a constructor functions, such as :func:`mlx_array_new()`, and one
need to free the allocated object through a corresponding free call, for
example :func:`mlx_array_free()`. There should be always one single
``free`` call corresponding to a given ``new`` call.

Once an object is created, one can perform multiple assignments, either
through ``set`` functions (such as :func:`mlx_array_set()`), or through MLX
operations. For example, the following code is perfectly valid in MLX C:

.. code-block:: c

    mlx_stream stream = mlx_default_gpu_stream_new();
    mlx_array a = mlx_array_new_float(1.0);
    mlx_array b = mlx_array_new_float(1.0);
    mlx_array_add(&b, a, b, stream); // b now holds a+b=2
    mlx_array_add(&b, a, b, stream); // b now holds 3
    mlx_array_set(&a, b); // a now holds 3 too
    mlx_array_free(a);
    mlx_array_free(b);

Apart few convenience functions returning information on each object, most
MLX C operations return values by argument. Return values will be passed
through the first pointer arguments of each function call.

Array
-----

The most important object in MLX C is certainly the :doc:`array <array>`
(:class:`mlx_array`), which holds the data on which computations are
performed. As MLX is `lazy
<https://ml-explore.github.io/mlx/build/html/usage/lazy_evaluation.html>`_,
the contents of the array obtained via the :func:`mlx_array_data_*()` functions are
valid only if :func:`mlx_eval()` as been called (see
:doc:`transforms <transforms>`).

Vector of Arrays, and Vector of Vector of Arrays
------------------------------------------------

MLX defines several types of :doc:`vectors <vector>`, including vector of
arrays (:class:`mlx_vector_array`) which can hold multiple arrays, and
vector of vector arrays (:class:`mlx_vector_vector_array`) which can hold
multiple vector of arrays.

An array added to a :class:`mlx_vector_array` will stay alive until the
vector of arrays is destroyed (via :func:`mlx_vector_array_free()`).

Same idea applies to :class:`mlx_vector_vector_array`, or other types of
arrays.


Device and Stream
-----------------

In MLX, arrays are not tied to a device. Instead, operations on arrays are
scheduled on a :doc:`stream <stream>`, which is associated to a particular
:doc:`device <device>`.

MLX C provides :class:`MLX_CPU_STREAM` and
:class:`MLX_GPU_STREAM`, which point to the default CPU and GPU
streams. See the basic `MLX C example
<https://github.com/ml-explore/mlx-c/blob/main/examples/example.c>`_.

String and Maps
---------------

MLX C has a :class:`mlx_string` which :doc:`encapsulates a C char
pointer <string>`. Just like other MLX C objects, it must be freed with 
:func:`mlx_string_free()`.

MLX C also has a :doc:`string-to-array map <map>` named
:class:`mlx_map_string_to_array`.

Array Operations
----------------

Many array :doc:`operations <ops>` are available, with additional support
for :doc:`random number generation <random>`, and :doc:`FFTs <fft>`. Advanced
:doc:`linear algebra operations <linalg>` are in their early stages.

IO Operations
-------------

MLX C wraps a number of :doc:`array IO operations <io>`, which save and
load arrays in several common formats.


Function Transformations
------------------------

MLX supports the concept of `function transforms
<https://ml-explore.github.io/mlx/build/html/python/transforms.html>`_.

These are also available in MLX C through the use of :doc:`closures
<closure>` that contain a C function pointer and optional
payloads. Closures obey the same memory management rules as other MLX C
objects and must be released with a matching ``free`` call (such as
:func:`mlx_closure_free()`).

MLX C :doc:`transforms <transforms>` will are applied on closures and may
return closures.

For more details, see the `basic closure example
<https://github.com/ml-explore/mlx-c/blob/main/examples/example-closure.c>`_,
or the `example to compute gradients
<https://github.com/ml-explore/mlx-c/blob/main/examples/example-grad.c>`_.

Compilation
-----------

When using the same function multiple times, compilation may be beneficial.
Compiling functions makes them more efficient by reducing redundant work,
fusing kernels, and reducing overhead. :doc:`Compilation operations <compile>`
are function transformations which take a closure and return a new closure
(which is the compiled version of the given closure).

Fast Custom Ops
===============

To maximize performance MLX has :doc:`fast <fast>` custom implementations
for some common operations.

Metal Backend-specific Functions
================================

MLX C exposes some useful functions related to the MLX :doc:`Metal backend
<metal>`.

Error Management
================

Most of MLX operations return an ``int`` value, which will be zero if the
operation was successful, or non-zero if some error occurred.

However, by default, the program will exit when an error occurs: each time
an error is encountered, the MLX C :doc:`error handler <error>` is called,
and the default error handler will simply print out the error, then exit.

It is possible to override the MLX C error handler, via the
:func:`mlx_set_error_handler()` function. Passing a ``NULL`` pointer to
this function will also reset the error handler to the default one.

That way, one may install a no-op error handler and then check each
function return value by hand, or adapt the error handler to an appropriate
behavior when embedding MLX C in another language.
