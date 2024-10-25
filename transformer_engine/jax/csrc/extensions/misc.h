/*************************************************************************
 * Copyright (c) 2022-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 *
 * See LICENSE for license information.
 ************************************************************************/

#include <transformer_engine/transformer_engine.h>

#include <cassert>
#include <string>
#include <vector>

namespace transformer_engine {
namespace jax {

constexpr int kMaxNumDim = 8;

struct Shape {
  int num_dim;
  size_t dims[kMaxNumDim];

  void from_vector(const std::vector<size_t> &shape);

  std::vector<size_t> to_vector() const;
};

std::vector<size_t> MakeShapeVector(NVTEShape shape);

struct Attn_Attr_Type {
  int64_t input_batch;
  int64_t bias_batch;
  int64_t q_max_seqlen;
  int64_t kv_max_seqlen;
  int64_t attn_heads;
  int64_t num_gqa_groups;
  int64_t bias_heads;
  int64_t head_dim;
  int64_t max_segments_per_seq;
  int64_t window_size_left;
  int64_t window_size_right;
  int64_t bias_type;
  int64_t mask_type;
  int64_t qkv_layout;
  double scaling_factor;
  double dropout_probability;
  bool is_training;
  bool deterministic;
};

}  // namespace jax
}  // namespace transformer_engine
