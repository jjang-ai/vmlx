#ifndef MLX_IO_PRIVATE_H
#define MLX_IO_PRIVATE_H

#include <iostream>
#include "mlx/mlx.h"

namespace {

class CReader : public mlx::core::io::Reader {
 public:
  void* desc;
  mlx_io_vtable vtable;

  CReader(void* desc, mlx_io_vtable vtable) : desc(desc), vtable(vtable) {};
  virtual bool is_open() const override {
    return vtable.is_open(desc);
  };
  virtual bool good() const override {
    return vtable.good(desc);
  };
  virtual size_t tell() override {
    return vtable.tell(desc);
  }
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) override {
    switch (way) {
      case std::ios_base::beg:
        return vtable.seek(desc, off, SEEK_SET);
        break;
      case std::ios_base::cur:
        return vtable.seek(desc, off, SEEK_CUR);
        break;
      case std::ios_base::end:
        return vtable.seek(desc, off, SEEK_END);
        break;
      default:
        throw std::runtime_error("mlx_io_reader: invalid seek way");
    }
  }
  virtual void read(char* data, size_t n) override {
    return vtable.read(desc, data, n);
  };
  virtual void read(char* data, size_t n, size_t offset) override {
    return vtable.read_at_offset(desc, data, n, offset);
  };
  virtual std::string label() const override {
    return vtable.label(desc);
  };
  virtual ~CReader() {
    vtable.free(desc);
  }
};

class CWriter : public mlx::core::io::Writer {
 public:
  void* desc;
  mlx_io_vtable vtable;

  CWriter(void* desc, mlx_io_vtable vtable) : desc(desc), vtable(vtable) {};
  virtual bool is_open() const override {
    return vtable.is_open(desc);
  };
  virtual bool good() const override {
    return vtable.good(desc);
  };
  virtual size_t tell() override {
    return vtable.tell(desc);
  }
  virtual void seek(
      int64_t off,
      std::ios_base::seekdir way = std::ios_base::beg) override {
    switch (way) {
      case std::ios_base::beg:
        return vtable.seek(desc, off, SEEK_SET);
        break;
      case std::ios_base::cur:
        return vtable.seek(desc, off, SEEK_CUR);
        break;
      case std::ios_base::end:
        return vtable.seek(desc, off, SEEK_END);
        break;
      default:
        throw std::runtime_error("mlx_io_writer: invalid seek way");
    }
  }
  virtual void write(const char* data, size_t n) override {
    return vtable.write(desc, data, n);
  };
  virtual std::string label() const override {
    return vtable.label(desc);
  };
  virtual ~CWriter() {
    vtable.free(desc);
  }
};

struct creader_holder {
  std::shared_ptr<CReader> ptr;
};

inline mlx_io_reader mlx_io_reader_new_(void* uctx, mlx_io_vtable vtable) {
  return mlx_io_reader(
      {new creader_holder({std::make_shared<CReader>(uctx, vtable)})});
}

inline std::shared_ptr<CReader> mlx_io_reader_get_(mlx_io_reader d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_io_reader");
  }
  return static_cast<creader_holder*>(d.ctx)->ptr;
}

inline void mlx_io_reader_free_(mlx_io_reader io) {
  if (io.ctx) {
    delete static_cast<creader_holder*>(io.ctx);
  }
}

struct cwriter_holder {
  std::shared_ptr<CWriter> ptr;
};

inline mlx_io_writer mlx_io_writer_new_(void* uctx, mlx_io_vtable vtable) {
  return mlx_io_writer(
      {new cwriter_holder({std::make_shared<CWriter>(uctx, vtable)})});
}

inline std::shared_ptr<CWriter> mlx_io_writer_get_(mlx_io_writer d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_io_writer");
  }
  return static_cast<cwriter_holder*>(d.ctx)->ptr;
}

inline void mlx_io_writer_free_(mlx_io_writer io) {
  if (io.ctx) {
    delete static_cast<cwriter_holder*>(io.ctx);
  }
}

} // namespace

#endif
