#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <string>

#include <torch/extension.h>

#include "csrc/bf16/mega_moe_bf16_result.h"

struct MegaMoeTilingData;

namespace deepep::megamoe_train::ops {

MegaMoeBF16ForwardResult MegaMoeBF16(torch::Tensor context, torch::Tensor x, torch::Tensor topk_ids,
                                     torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                     int64_t max_recv_tokens, int rank, int world_size,
                                     std::size_t symmetric_buffer_bytes, uint64_t ffts_addr, void* symmetric_buffer,
                                     const std::string& save_gmm1_output = "bf16");

// Training forward ABI returns
// (y, expert_token_nums, training_saved_state, dispatch_output,
//  forward_transient_workspace). The framework chooses whether to retain the
// independent dispatch output for backward; transient scratch is never saved.
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MegaMoeBF16Forward(
    torch::Tensor context, torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights, torch::Tensor weight1,
    torch::Tensor weight2, int64_t max_recv_tokens, int rank, int world_size, std::size_t symmetric_buffer_bytes,
    uint64_t ffts_addr, void* symmetric_buffer, double swiglu_limit = 0.0,
    const std::string& save_gmm1_output = "bf16");

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> MegaMoeBF16Backward(
    torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights, torch::Tensor weight1,
    torch::Tensor weight2, torch::Tensor expert_token_nums, torch::Tensor training_saved_state,
    torch::Tensor dispatch_output, int64_t max_recv_tokens, int rank, int world_size,
    std::size_t symmetric_buffer_bytes, uint64_t ffts_addr, void* symmetric_buffer, double swiglu_limit = 0.0,
    const std::string& save_gmm1_output = "bf16");

}  // namespace deepep::megamoe_train::ops
