#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

#include <torch/extension.h>

struct MegaMoeTilingData;

namespace deepep::megamoe_train::ops {

struct ForwardResult {
    torch::Tensor y;
    torch::Tensor counts;
    torch::Tensor saved_state;
    torch::Tensor dispatch_output;
};

struct BackwardResult {
    torch::Tensor dx;
    torch::Tensor dtopk_weights;
    torch::Tensor dw1;
    torch::Tensor dw2;
};

ForwardResult MegaMoeBF16Forward(torch::Tensor context, torch::Tensor x, torch::Tensor topk_ids,
                                 torch::Tensor topk_weights, torch::Tensor weight1, torch::Tensor weight2,
                                 int64_t max_recv_tokens, int rank, int world_size, std::size_t symmetric_buffer_bytes,
                                 uint64_t ffts_addr, void* symmetric_buffer, double swiglu_limit = 0.0,
                                 const std::string& save_gmm1_output = "bf16");

BackwardResult MegaMoeBF16Backward(torch::Tensor context, torch::Tensor grad_y, torch::Tensor topk_weights,
                                   torch::Tensor weight1, torch::Tensor weight2, torch::Tensor expert_token_nums,
                                   torch::Tensor training_saved_state, torch::Tensor dispatch_output,
                                   int64_t max_recv_tokens, int rank, int world_size,
                                   std::size_t symmetric_buffer_bytes, uint64_t ffts_addr, void* symmetric_buffer,
                                   double swiglu_limit = 0.0, const std::string& save_gmm1_output = "bf16");

}  // namespace deepep::megamoe_train::ops
