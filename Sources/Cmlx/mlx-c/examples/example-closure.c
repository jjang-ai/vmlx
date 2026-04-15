/* Copyright © 2023-2024 Apple Inc. */

#include <math.h>
#include <stdio.h>
#include "mlx/c/mlx.h"

void print_array(const char* msg, mlx_array arr) {
  mlx_string str = mlx_string_new();
  mlx_array_tostring(&str, arr);
  printf("%s\n%s\n", msg, mlx_string_data(str));
  mlx_string_free(str);
}

int inc_fun(mlx_array* res_, mlx_array in) {
  mlx_stream stream = mlx_default_gpu_stream_new();
  mlx_array value = mlx_array_new_float(1.0);
  mlx_add(res_, in, value, stream);
  mlx_stream_free(stream);
  mlx_array_free(value);
  return 0;
}

struct bogus_payload {
  mlx_array value;
  char error[256];
};

int inc_fun_bogus(
    mlx_vector_array* vres_,
    mlx_vector_array in,
    void* payload_) {
  struct bogus_payload* payload = payload_;
  mlx_stream stream = mlx_default_gpu_stream_new();
  if (mlx_vector_array_size(in) != 1) {
    fprintf(stderr, "inc_func_value: expected 1 argument");
    exit(EXIT_FAILURE);
  }

  // check if there is NaN in payload value
  bool has_nan_flag;
  mlx_array value = payload->value;
  mlx_array has_nan = mlx_array_new();
  mlx_isnan(&has_nan, value, stream);
  mlx_any(&has_nan, has_nan, false, stream);
  mlx_array_item_bool(&has_nan_flag, has_nan);
  mlx_array_free(has_nan);

  if (has_nan_flag) {
    mlx_stream_free(stream);
    snprintf(payload->error, 256, "nan detected");
    return 1;
  }

  mlx_array res = mlx_array_new();
  mlx_vector_array_get(&res, in, 0);
  mlx_add(&res, res, value, stream);
  mlx_vector_array_set_value(vres_, res);
  mlx_array_free(res);
  mlx_stream_free(stream);
  return 0;
}

void error_handler_noop(const char* msg, void* data) {
  printf("ignoring the error: <%s>\n", msg);
}

int main() {
  mlx_array x = mlx_array_new_float(1.0);
  print_array("x: ", x);

  // simple +1 to input
  mlx_array y = mlx_array_new();
  mlx_vector_array v_y = mlx_vector_array_new();
  mlx_vector_array v_x = mlx_vector_array_new_value(x);
  mlx_closure cls = mlx_closure_new_unary(inc_fun);
  mlx_closure_apply(&v_y, cls, v_x);
  mlx_vector_array_get(&y, v_y, 0);
  print_array("+1: ", y);

  struct bogus_payload payload;

  // simple +2 to input, with paylaod
  payload.value = mlx_array_new_float(2.0);
  mlx_closure cls_with_value =
      mlx_closure_new_func_payload(inc_fun_bogus, &payload, NULL);
  mlx_closure_apply(&v_y, cls_with_value, v_x);
  mlx_vector_array_get(&y, v_y, 0);
  print_array("+2: ", y);

  // simple +nan to input, with paylaod
  mlx_set_error_handler(error_handler_noop, NULL, NULL);
  mlx_array_set_float(&payload.value, NAN);
  if (mlx_closure_apply(&v_y, cls_with_value, v_x)) {
    printf("closure failed with: <%s>\n", payload.error);
  } else {
    mlx_vector_array_get(&y, v_y, 0);
    print_array("+nan: ", y);
  }
  mlx_set_error_handler(NULL, NULL, NULL);

  mlx_array_free(x);
  mlx_array_free(y);
  mlx_array_free(payload.value);
  mlx_vector_array_free(v_x);
  mlx_vector_array_free(v_y);
  mlx_closure_free(cls);
  mlx_closure_free(cls_with_value);

  return 0;
}
