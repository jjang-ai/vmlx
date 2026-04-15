/* Copyright © 2023-2024 Apple Inc.                   */
/*                                                    */
/* This file is auto-generated. Do not edit manually. */
/*                                                    */

#ifndef MLX_EXPORT_PRIVATE_H
#define MLX_EXPORT_PRIVATE_H

#include "mlx/c/export.h"
#include "mlx/mlx.h"

inline mlx_function_exporter mlx_function_exporter_new_() {
  return mlx_function_exporter({nullptr});
}

inline mlx_function_exporter mlx_function_exporter_new_(
    mlx::core::FunctionExporter&& s) {
  return mlx_function_exporter({new mlx::core::FunctionExporter(std::move(s))});
}

inline mlx_function_exporter& mlx_function_exporter_set_(
    mlx_function_exporter& d,
    mlx::core::FunctionExporter&& s) {
  if (d.ctx) {
    delete static_cast<mlx::core::FunctionExporter*>(d.ctx);
  }
  d.ctx = new mlx::core::FunctionExporter(std::move(s));
  return d;
}

inline mlx::core::FunctionExporter& mlx_function_exporter_get_(
    mlx_function_exporter d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_function_exporter");
  }
  return *static_cast<mlx::core::FunctionExporter*>(d.ctx);
}

inline void mlx_function_exporter_free_(mlx_function_exporter d) {
  if (d.ctx) {
    delete static_cast<mlx::core::FunctionExporter*>(d.ctx);
  }
}

inline mlx_imported_function mlx_imported_function_new_() {
  return mlx_imported_function({nullptr});
}

inline mlx_imported_function mlx_imported_function_new_(
    mlx::core::ImportedFunction&& s) {
  return mlx_imported_function({new mlx::core::ImportedFunction(std::move(s))});
}

inline mlx_imported_function& mlx_imported_function_set_(
    mlx_imported_function& d,
    mlx::core::ImportedFunction&& s) {
  if (d.ctx) {
    delete static_cast<mlx::core::ImportedFunction*>(d.ctx);
  }
  d.ctx = new mlx::core::ImportedFunction(std::move(s));
  return d;
}

inline mlx::core::ImportedFunction& mlx_imported_function_get_(
    mlx_imported_function d) {
  if (!d.ctx) {
    throw std::runtime_error("expected a non-empty mlx_imported_function");
  }
  return *static_cast<mlx::core::ImportedFunction*>(d.ctx);
}

inline void mlx_imported_function_free_(mlx_imported_function d) {
  if (d.ctx) {
    delete static_cast<mlx::core::ImportedFunction*>(d.ctx);
  }
}

#endif
