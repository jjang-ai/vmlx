#include "mlx/c/export.h"
#include "mlx/c/error.h"
#include "mlx/c/private/mlx.h"
#include "mlx/export.h"

extern "C" int mlx_export_function(
    const char* file,
    const mlx_closure fun,
    const mlx_vector_array args,
    bool shapeless) {
  try {
    mlx::core::export_function(
        std::string(file),
        mlx_closure_get_(fun),
        mlx_vector_array_get_(args),
        shapeless);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_export_function_kwargs(
    const char* file,
    const mlx_closure_kwargs fun,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs,
    bool shapeless) {
  try {
    mlx::core::export_function(
        std::string(file),
        mlx_closure_kwargs_get_(fun),
        mlx_vector_array_get_(args),
        mlx_map_string_to_array_get_(kwargs),
        shapeless);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" mlx_function_exporter mlx_function_exporter_new(
    const char* file,
    const mlx_closure fun,
    bool shapeless) {
  try {
    return mlx_function_exporter_new_(
        mlx::core::exporter(
            std::string(file), mlx_closure_get_(fun), shapeless));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return {nullptr};
  }
}
extern "C" int mlx_function_exporter_free(mlx_function_exporter xfunc) {
  try {
    mlx_function_exporter_free_(xfunc);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_function_exporter_apply(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args) {
  try {
    mlx_function_exporter_get_(xfunc)(mlx_vector_array_get_(args));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}

extern "C" int mlx_function_exporter_apply_kwargs(
    const mlx_function_exporter xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs) {
  try {
    mlx_function_exporter_get_(xfunc)(
        mlx_vector_array_get_(args), mlx_map_string_to_array_get_(kwargs));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" mlx_imported_function mlx_imported_function_new(const char* file) {
  try {
    return mlx_imported_function_new_(
        mlx::core::import_function(std::string(file)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return {nullptr};
  }
}
extern "C" int mlx_imported_function_free(mlx_imported_function xfunc) {
  try {
    mlx_imported_function_free_(xfunc);
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_imported_function_apply(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args) {
  try {
    mlx_vector_array_set_(
        *res, mlx_imported_function_get_(xfunc)(mlx_vector_array_get_(args)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
extern "C" int mlx_imported_function_apply_kwargs(
    mlx_vector_array* res,
    const mlx_imported_function xfunc,
    const mlx_vector_array args,
    const mlx_map_string_to_array kwargs) {
  try {
    mlx_vector_array_set_(
        *res,
        mlx_imported_function_get_(xfunc)(
            mlx_vector_array_get_(args), mlx_map_string_to_array_get_(kwargs)));
  } catch (std::exception& e) {
    mlx_error(e.what());
    return 1;
  }
  return 0;
}
