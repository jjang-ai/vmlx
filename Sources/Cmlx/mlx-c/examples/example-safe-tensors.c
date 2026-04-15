/* Copyright © 2023-2024 Apple Inc. */

#include <stdio.h>
#include <string.h>
#include "mlx/c/mlx.h"

void print_array(const char* msg, mlx_array arr) {
  mlx_string str = mlx_string_new();
  mlx_array_tostring(&str, arr);
  printf("%s\n%s\n", msg, mlx_string_data(str));
  mlx_string_free(str);
}

typedef struct mlx_mem_stream_ {
  char* data;
  size_t pos;
  size_t size;
  bool err;
  bool free_data;
} mlx_mem_stream;
bool mem_is_open(void* desc) {
  printf("ISOPEN\n");
  return desc != NULL;
}
bool mem_good(void* desc) {
  printf("GOOD\n");
  mlx_mem_stream* m_desc = desc;
  return !m_desc->err;
}
size_t mem_tell(void* desc) {
  printf("TELL\n");
  mlx_mem_stream* m_desc = desc;
  return m_desc->pos;
}
void mem_seek(void* desc, int64_t off, int whence) {
  printf("SEEK\n");
  mlx_mem_stream* m_desc = desc;
  size_t new_pos;
  switch (whence) {
    case SEEK_SET:
      new_pos = off;
      break;
    case SEEK_CUR:
      new_pos = m_desc->pos + off;
      break;
    case SEEK_END:
      new_pos = m_desc->size + off;
      break;
    default:
      m_desc->err = true;
      return;
  }
  if (new_pos > m_desc->size) {
    m_desc->err = true;
  } else {
    m_desc->pos = new_pos;
  }
}
void mem_read(void* desc, char* data, size_t n) {
  printf("READ %ld\n", n);
  mlx_mem_stream* m_desc = desc;
  if (n + m_desc->pos > m_desc->size) {
    m_desc->err = true;
    return;
  }
  memcpy(data, m_desc->data + m_desc->pos, n);
  m_desc->pos += n;
}
void mem_read_at_offset(void* desc, char* data, size_t n, size_t off) {
  printf("READ@OFFSET %ld @ %ld\n", n, off);
  mlx_mem_stream* m_desc = desc;
  if (off + n > m_desc->size) {
    m_desc->err = true;
    return;
  }
  memcpy(data, m_desc->data + off, n);
  m_desc->pos = off;
}
void mem_write(void* desc, const char* data, size_t n) {
  printf("WRITE %ld\n", n);
  mlx_mem_stream* m_desc = desc;
  if (n + m_desc->pos > m_desc->size) {
    m_desc->err = true;
    return;
  }
  memcpy(m_desc->data + m_desc->pos, data, n);
  m_desc->pos += n;
}
const char* mem_label(void* desc) {
  printf("LABEL\n");
  return "<custom memory stream>";
}
void mem_free(void* desc) {
  mlx_mem_stream* m_desc = desc;
  if (m_desc->free_data) {
    printf("FREE DATA\n");
    free(m_desc->data);
  }
}
static mlx_io_vtable mlx_io_vtable_mlx_mem_stream = {
    &mem_is_open,
    &mem_good,
    &mem_tell,
    &mem_seek,
    &mem_read,
    &mem_read_at_offset,
    &mem_write,
    &mem_label,
    &mem_free};

int main(void) {
  mlx_stream stream = mlx_default_cpu_stream_new();
  mlx_map_string_to_array data = mlx_map_string_to_array_new();
  mlx_map_string_to_string metadata = mlx_map_string_to_string_new();

  printf("load data from disk:\n");
  mlx_load_safetensors(&data, &metadata, "arrays.safetensors", stream);
  mlx_map_string_to_array_iterator it =
      mlx_map_string_to_array_iterator_new(data);
  const char* key;
  mlx_array value = mlx_array_new();
  while (!mlx_map_string_to_array_iterator_next(&key, &value, it)) {
    print_array(key, value);
  }

  printf("attempting to write arrays in a memory stream\n");
  mlx_mem_stream mem_stream = {
      malloc(2048), // 2048 bytes
      0L, // position
      2048L, // size
      false, // err
      false // do not free data (we will reuse it at read time)
  };
  mlx_io_writer writer =
      mlx_io_writer_new(&mem_stream, mlx_io_vtable_mlx_mem_stream);
  mlx_save_safetensors_writer(writer, data, metadata);
  mlx_io_writer_free(writer);

  printf(
      "position in memory stream: %ld err flag: %d\n",
      mem_stream.pos,
      mem_stream.err);
  printf("data in memory stream: ");
  for (size_t i = 0; i < mem_stream.pos; i++) {
    printf("%c", mem_stream.data[i]);
  }
  printf("\n");

  // reinit everything
  mem_stream.pos = 0L;
  mlx_map_string_to_array_free(data);
  mlx_map_string_to_string_free(metadata);
  mlx_map_string_to_array_iterator_free(it);

  printf("attempting to read from memory\n");
  mem_stream.free_data = true;
  mlx_io_reader reader =
      mlx_io_reader_new(&mem_stream, mlx_io_vtable_mlx_mem_stream);
  data = mlx_map_string_to_array_new();
  metadata = mlx_map_string_to_string_new();
  mlx_load_safetensors_reader(&data, &metadata, reader, stream);
  mlx_io_reader_free(reader);

  printf("now the arrays (lazily evaluated):\n");
  it = mlx_map_string_to_array_iterator_new(data);
  while (!mlx_map_string_to_array_iterator_next(&key, &value, it)) {
    print_array(key, value);
  }

  mlx_array_free(value);
  mlx_map_string_to_array_free(data);
  mlx_map_string_to_string_free(metadata);
  mlx_map_string_to_array_iterator_free(it);
  mlx_stream_free(stream);
  return 0;
}
