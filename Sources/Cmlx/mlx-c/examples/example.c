/* Copyright © 2023-2024 Apple Inc. */

#include <stdio.h>
#include "mlx/c/mlx.h"

void print_array(const char* msg, mlx_array arr) {
  mlx_string str = mlx_string_new();
  mlx_array_tostring(&str, arr);
  printf("%s\n%s\n", msg, mlx_string_data(str));
  mlx_string_free(str);
}

void gpu_info(void) {
  printf("==================================================\n");
  printf("GPU info (using mlx_device_info API):\n");

  // Get device count
  int gpu_count = 0;
  mlx_device_count(&gpu_count, MLX_GPU);
  printf("GPU device count: %d\n", gpu_count);

  // Get default device
  mlx_device dev = mlx_device_new();
  mlx_get_default_device(&dev);

  // Get device info using new API
  mlx_device_info info = mlx_device_info_new();
  if (mlx_device_info_get(&info, dev) == 0) {
    // Get all available keys
    mlx_vector_string keys = mlx_vector_string_new();
    mlx_device_info_get_keys(&keys, info);
    size_t num_keys = mlx_vector_string_size(keys);

    printf("Device info (%zu keys):\n", num_keys);
    for (size_t i = 0; i < num_keys; i++) {
      char* key = NULL;
      mlx_vector_string_get(&key, keys, i);
      if (!key)
        continue;

      bool is_string = false;
      mlx_device_info_is_string(&is_string, info, key);

      if (is_string) {
        const char* value = NULL;
        mlx_device_info_get_string(&value, info, key);
        printf("  %s: %s\n", key, value ? value : "(null)");
      } else {
        size_t value = 0;
        mlx_device_info_get_size(&value, info, key);
        printf("  %s: %zu\n", key, value);
      }
    }

    mlx_vector_string_free(keys);
  }

  mlx_device_info_free(info);
  mlx_device_free(dev);

  printf("==================================================\n");
}

int main(void) {
  mlx_string version = mlx_string_new();
  mlx_version(&version);
  printf("MLX version: %s\n", mlx_string_data(version));

  gpu_info();

  // Use default CPU stream (works with any backend)
  mlx_stream stream = mlx_default_cpu_stream_new();
  float data[] = {1, 2, 3, 4, 5, 6};
  int shape[] = {2, 3};
  mlx_array arr = mlx_array_new_data(data, shape, 2, MLX_FLOAT32);
  print_array("hello world!", arr);

  mlx_array two = mlx_array_new_int(2);
  mlx_divide(&arr, arr, two, stream);
  print_array("divide by 2!", arr);

  mlx_arange(&arr, 0, 3, 0.5, MLX_FLOAT32, stream);
  print_array("arange", arr);

  float* data_managed = malloc(6 * sizeof(float));
  for (int i = 0; i < 6; i++) {
    data_managed[i] = (float)i;
  }
  mlx_array arr_managed =
      mlx_array_new_data_managed(data_managed, shape, 2, MLX_FLOAT32, free);
  print_array("from user buffer", arr_managed);

  mlx_array_free(arr);
  mlx_array_free(two);
  mlx_array_free(arr_managed);
  mlx_stream_free(stream);
  mlx_string_free(version);

  return 0;
}
