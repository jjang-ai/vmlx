/* Copyright © 2023-2025 Apple Inc. */

#include <math.h>
#include <stdio.h>
#include "mlx/c/mlx.h"

void print_array(const char* msg, mlx_array arr) {
  mlx_string str = mlx_string_new();
  mlx_array_tostring(&str, arr);
  printf("%s\n%s\n", msg, mlx_string_data(str));
  mlx_string_free(str);
}

int inc_fun(mlx_array* res_, mlx_array args) {
  mlx_stream stream = mlx_default_gpu_stream_new();
  mlx_array value = mlx_array_new_float(1.0);
  mlx_add(res_, args, value, stream);
  mlx_stream_free(stream);
  mlx_array_free(value);
  return 0;
}

int mul_fun(
    mlx_vector_array* res_,
    mlx_vector_array args,
    mlx_map_string_to_array kwargs) {
  mlx_stream stream = mlx_default_gpu_stream_new();
  mlx_array x = mlx_array_new();
  mlx_array y = mlx_array_new();
  mlx_array res = mlx_array_new();

  mlx_map_string_to_array_get(&x, kwargs, "x");
  mlx_map_string_to_array_get(&y, kwargs, "y");
  mlx_multiply(&res, x, y, stream);
  mlx_vector_array_set_value(res_, res);

  mlx_array_free(res);
  mlx_array_free(y);
  mlx_array_free(x);
  mlx_stream_free(stream);

  return 0;
}

int main(void) {
  mlx_array x = mlx_array_new_float(1.0);
  print_array("x: ", x);

  printf("storing inc() function in inc_func.bin file\n");
  mlx_vector_array args = mlx_vector_array_new_value(x);
  mlx_closure cls = mlx_closure_new_unary(inc_fun);
  mlx_export_function("inc_func.bin", cls, args, false);
  mlx_closure_free(cls);

  printf("loading inc() function from inc_func.bin file\n");
  mlx_imported_function xfunc_inc = mlx_imported_function_new("inc_func.bin");

  printf("evaluating inc() over x\n");
  mlx_vector_array res = mlx_vector_array_new();
  mlx_imported_function_apply(&res, xfunc_inc, args);

  mlx_array y = mlx_array_new();
  mlx_vector_array_get(&y, res, 0);
  print_array("+1: ", y);
  mlx_array_set(&x, y);

  printf("evaluating inc() over x with kwargs\n");
  mlx_vector_array empty_args = mlx_vector_array_new();
  mlx_map_string_to_array kwargs = mlx_map_string_to_array_new();
  mlx_map_string_to_array_insert(kwargs, "x", x);
  mlx_imported_function_apply_kwargs(&res, xfunc_inc, empty_args, kwargs);
  mlx_vector_array_get(&y, res, 0);
  print_array("+1: ", y);
  mlx_array_set(&x, y);

  printf("storing mul() function in mul_func.bin file\n");
  mlx_map_string_to_array_insert(kwargs, "y", x);
  mlx_closure_kwargs cls_kwargs = mlx_closure_kwargs_new_func(mul_fun);
  mlx_export_function_kwargs(
      "mul_func.bin", cls_kwargs, empty_args, kwargs, false);
  mlx_closure_kwargs_free(cls_kwargs);

  printf("loading mul() function from mul_func.bin file\n");
  mlx_imported_function xfunc_mul = mlx_imported_function_new("mul_func.bin");
  printf("evaluating mul() over x and x with kwargs\n");
  print_array("x: ", x);
  mlx_map_string_to_array_insert(kwargs, "x", x);
  mlx_map_string_to_array_insert(kwargs, "y", x);
  mlx_imported_function_apply_kwargs(&res, xfunc_mul, empty_args, kwargs);
  mlx_vector_array_get(&y, res, 0);
  print_array("3*3: ", y);
  mlx_array_set(&x, y);

  mlx_array_free(y);
  mlx_vector_array_free(res);
  mlx_map_string_to_array_free(kwargs);
  mlx_vector_array_free(args);
  mlx_vector_array_free(empty_args);
  mlx_array_free(x);
  mlx_imported_function_free(xfunc_inc);
  mlx_imported_function_free(xfunc_mul);

  return 0;
}
