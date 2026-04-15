def __implement_mlx_fast_custom_kernel(backend, backend_specific_code, implementation):
    if implementation:
        code_config = """
struct mlx_fast_custom_kernel_config_cpp_ {
  std::vector<mlx::core::Shape> output_shapes;
  std::vector<mlx::core::Dtype> output_dtypes;
  std::tuple<int, int, int> grid;
  std::tuple<int, int, int> thread_group;
  std::vector<std::pair<std::string, mlx::core::fast::TemplateArg>>
      template_args;
  std::optional<float> init_value;
  bool verbose;
};

inline mlx_fast_custom_kernel_config mlx_fast_custom_kernel_config_new_() {
  return mlx_fast_custom_kernel_config(
      {new mlx_fast_custom_kernel_config_cpp_()});
}

inline mlx_fast_custom_kernel_config_cpp_& mlx_fast_custom_kernel_config_get_(
    mlx_fast_custom_kernel_config d) {
  if (!d.ctx) {
    throw std::runtime_error(
        "expected a non-empty mlx_fast_custom_kernel_config");
  }
  return *static_cast<mlx_fast_custom_kernel_config_cpp_*>(d.ctx);
}

inline void mlx_fast_custom_kernel_config_free_(mlx_fast_custom_kernel_config d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_custom_kernel_config_cpp_*>(d.ctx);
  }
}

extern "C" mlx_fast_custom_kernel_config mlx_fast_custom_kernel_config_new(void) {
  try {
    return mlx_fast_custom_kernel_config_new_();
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}

extern "C" void mlx_fast_custom_kernel_config_free(
    mlx_fast_custom_kernel_config cls) {
  mlx_fast_custom_kernel_config_free_(cls);
}

extern "C" int mlx_fast_custom_kernel_config_add_output_arg(
    mlx_fast_custom_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype) {
  try {
    mlx_fast_custom_kernel_config_get_(cls).output_shapes.push_back(
        mlx::core::Shape(shape, shape + size));
    mlx_fast_custom_kernel_config_get_(cls).output_dtypes.push_back(
        mlx_dtype_to_cpp(dtype));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_custom_kernel_config_set_grid(
    mlx_fast_custom_kernel_config cls,
    int grid1,
    int grid2,
    int grid3) {
  try {
    mlx_fast_custom_kernel_config_get_(cls).grid =
        std::make_tuple(grid1, grid2, grid3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_custom_kernel_config_set_thread_group(
    mlx_fast_custom_kernel_config cls,
    int thread1,
    int thread2,
    int thread3) {
  try {
    mlx_fast_custom_kernel_config_get_(cls).thread_group =
        std::make_tuple(thread1, thread2, thread3);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_custom_kernel_config_set_init_value(
    mlx_fast_custom_kernel_config cls,
    float value) {
  try {
    mlx_fast_custom_kernel_config_get_(cls).init_value = value;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_custom_kernel_config_set_verbose(
    mlx_fast_custom_kernel_config cls,
    bool verbose) {
  try {
    mlx_fast_custom_kernel_config_get_(cls).verbose = verbose;
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_custom_kernel_config_add_template_arg_dtype(
    mlx_fast_custom_kernel_config cls,
    const char* name,
    mlx_dtype dtype) {
  try {
    mlx_fast_custom_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), mlx_dtype_to_cpp(dtype)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_custom_kernel_config_add_template_arg_int(
    mlx_fast_custom_kernel_config cls,
    const char* name,
    int value) {
  try {
    mlx_fast_custom_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_fast_custom_kernel_config_add_template_arg_bool(
    mlx_fast_custom_kernel_config cls,
    const char* name,
    bool value) {
  try {
    mlx_fast_custom_kernel_config_get_(cls).template_args.push_back(
        std::make_pair(std::string(name), value));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
        """
        code_def = """
struct mlx_fast_custom_kernel_cpp_ {
  mlx::core::fast::CustomKernelFunction mkf;
  mlx_fast_custom_kernel_cpp_(mlx::core::fast::CustomKernelFunction mkf)
      : mkf(mkf) {};
};
        """

        code = """
inline mlx::core::fast::CustomKernelFunction& mlx_fast_custom_kernel_get_(
    mlx_fast_custom_kernel d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_fast_custom_kernel");
  }
  return static_cast<mlx_fast_custom_kernel_cpp_*>(d.ctx)->mkf;
}

inline void mlx_fast_custom_kernel_free_(mlx_fast_custom_kernel d) {
  if (d.ctx) {
    delete static_cast<mlx_fast_custom_kernel_cpp_*>(d.ctx);
  }
}

extern "C" void mlx_fast_custom_kernel_free(mlx_fast_custom_kernel cls) {
  mlx_fast_custom_kernel_free_(cls);
}

extern "C" int mlx_fast_custom_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_custom_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_custom_kernel_config config,
    const mlx_stream stream) {
  try {
    auto config_ctx = mlx_fast_custom_kernel_config_get_(config);
    mlx_vector_array_set_(
        *outputs,
        mlx_fast_custom_kernel_get_(cls)(
            mlx_vector_array_get_(inputs),
            config_ctx.output_shapes,
            config_ctx.output_dtypes,
            config_ctx.grid,
            config_ctx.thread_group,
            config_ctx.template_args,
            config_ctx.init_value,
            config_ctx.verbose,
            mlx_stream_get_(stream)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
        """
    else:
        code_config = """
typedef struct mlx_fast_custom_kernel_config_ {
  void* ctx;
} mlx_fast_custom_kernel_config;
mlx_fast_custom_kernel_config mlx_fast_custom_kernel_config_new(void);
void mlx_fast_custom_kernel_config_free(mlx_fast_custom_kernel_config cls);

int mlx_fast_custom_kernel_config_add_output_arg(
    mlx_fast_custom_kernel_config cls,
    const int* shape,
    size_t size,
    mlx_dtype dtype);
int mlx_fast_custom_kernel_config_set_grid(
    mlx_fast_custom_kernel_config cls,
    int grid1,
    int grid2,
    int grid3);
int mlx_fast_custom_kernel_config_set_thread_group(
    mlx_fast_custom_kernel_config cls,
    int thread1,
    int thread2,
    int thread3);
int mlx_fast_custom_kernel_config_set_init_value(
    mlx_fast_custom_kernel_config cls,
    float value);
int mlx_fast_custom_kernel_config_set_verbose(
    mlx_fast_custom_kernel_config cls,
    bool verbose);
int mlx_fast_custom_kernel_config_add_template_arg_dtype(
    mlx_fast_custom_kernel_config cls,
    const char* name,
    mlx_dtype dtype);
int mlx_fast_custom_kernel_config_add_template_arg_int(
    mlx_fast_custom_kernel_config cls,
    const char* name,
    int value);
int mlx_fast_custom_kernel_config_add_template_arg_bool(
    mlx_fast_custom_kernel_config cls,
    const char* name,
    bool value);
        """

        code_def = """
typedef struct mlx_fast_custom_kernel_ {
  void* ctx;
} mlx_fast_custom_kernel;
        """

        code = """
void mlx_fast_custom_kernel_free(mlx_fast_custom_kernel cls);

int mlx_fast_custom_kernel_apply(
    mlx_vector_array* outputs,
    mlx_fast_custom_kernel cls,
    const mlx_vector_array inputs,
    const mlx_fast_custom_kernel_config config,
    const mlx_stream stream);
        """

    code_config = code_config.replace("custom", backend)
    code_def = code_def.replace("custom", backend)
    code = code.replace("custom", backend)

    print(code_config)
    print(code_def)
    print(backend_specific_code)
    print(code)


def mlx_fast_cuda_kernel(f, implementation):
    if implementation:
        custom_code = """
inline mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    int shared_memory) {
  return mlx_fast_cuda_kernel(
      {new mlx_fast_cuda_kernel_cpp_(mlx::core::fast::cuda_kernel(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          shared_memory))});
}

extern "C" mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory) {
  try {
    return mlx_fast_cuda_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        shared_memory);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}
        """
    else:
        custom_code = """
mlx_fast_cuda_kernel mlx_fast_cuda_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    int shared_memory);
        """
    __implement_mlx_fast_custom_kernel("cuda", custom_code, implementation)


def mlx_fast_metal_kernel(f, implementation):
    if implementation:
        custom_code = """
inline mlx_fast_metal_kernel mlx_fast_metal_kernel_new_(
    const std::string& name,
    const std::vector<std::string>& input_names,
    const std::vector<std::string>& output_names,
    const std::string& source,
    const std::string& header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  return mlx_fast_metal_kernel(
      {new mlx_fast_metal_kernel_cpp_(mlx::core::fast::metal_kernel(
          name,
          input_names,
          output_names,
          source,
          header,
          ensure_row_contiguous,
          atomic_outputs))});
}

extern "C" mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs) {
  try {
    return mlx_fast_metal_kernel_new_(
        name,
        mlx_vector_string_get_(input_names),
        mlx_vector_string_get_(output_names),
        source,
        header,
        ensure_row_contiguous,
        atomic_outputs);
  } catch (std::exception& e) {
    mlx_error(e.what());
  }
  return {nullptr};
}
        """
    else:
        custom_code = """
mlx_fast_metal_kernel mlx_fast_metal_kernel_new(
    const char* name,
    const mlx_vector_string input_names,
    const mlx_vector_string output_names,
    const char* source,
    const char* header,
    bool ensure_row_contiguous,
    bool atomic_outputs);
        """
    __implement_mlx_fast_custom_kernel("metal", custom_code, implementation)
