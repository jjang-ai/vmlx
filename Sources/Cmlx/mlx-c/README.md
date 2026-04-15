# MLX C

MLX C is a C API for [MLX](https://github.com/ml-explore/mlx).

MLX is an array framework for machine learning on Apple silicon. MLX C expands
MLX to the C language, making research and experimentation easier on Apple
silicon.

MLX C can be used standalone or as a bridge to bind other languages to
MLX. For example, the [MLX Swift](https://github.com/ml-explore/mlx-swift/)
package uses MLX C to provide a Swift API to MLX.

For more information see the [docs](https://ml-explore.github.io/mlx-c).

## Install

CMake is required to build MLX C. You can install it with [Homebrew](https://brew.sh/):

```shell
brew install cmake
```

To build, run the following commands:

```shell
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

From the `build/` directory, you can run an [example](examples/example.c)
that uses MLX C with `./example`.

## Contributing

Check out the [contribution guidelines](CONTRIBUTING.md) for more information
on contributing to MLX C. See the
[docs](https://ml-explore.github.io/mlx/build/html/install.html) for more
information on building from source, and running tests.

We are grateful for all of [our
contributors](ACKNOWLEDGMENTS.md#Individual-Contributors). If you contribute
to MLX C and wish to be acknowledged, please add your name to the list in your
pull request.
