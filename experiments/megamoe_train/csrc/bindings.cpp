#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>

#include <acl/acl.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>

#include "shmem.h"

#include "csrc/bf16/check.h"
#include "csrc/bf16/launcher.h"
#include "csrc/bf16/mega_moe.h"
#include "csrc/bf16/mega_moe_bf16.h"

namespace deepep::megamoe_train {
namespace {

constexpr std::size_t kDeviceLargePageBytes = 2 * 1024 * 1024;
constexpr int64_t kMegaMoeContextBytes = static_cast<int64_t>(sizeof(uint32_t) * 2 + sizeof(uint64_t) * 1025);

}  // namespace

// This object borrows a checked root Session window; it never initializes,
// allocates or finalizes SHMEM. Python owns the Session until explicit close.
class BF16Executor {
public:
    BF16Executor(uintptr_t address, int64_t bytes, int rank, int world_size, torch::Tensor like)
        : address_(address), bytes_(bytes), rank_(rank), world_size_(world_size)
    {
        TORCH_CHECK(address != 0 && bytes > 0 && bytes % kDeviceLargePageBytes == 0,
                    "invalid borrowed training window");
        TORCH_CHECK(rank >= 0 && rank < world_size && world_size <= 1024, "invalid training rank/world size");
        TORCH_CHECK(aclshmem_my_pe() == rank && aclshmem_n_pes() == world_size,
                    "training identity disagrees with the root Session");
        ffts_addr_ = shmemx_get_ffts_config();
        (void)Context(like);
    }

    void close()
    {
        address_ = 0;
        context_ = torch::Tensor();
    }

    ops::ForwardResult forward(torch::Tensor x, torch::Tensor topk_ids, torch::Tensor topk_weights,
                               torch::Tensor weight1, torch::Tensor weight2, int64_t max_recv_tokens,
                               double swiglu_limit, const std::string& save_gmm1_output)
    {
        TORCH_CHECK(address_ != 0, "training operator has been closed");
        return ops::MegaMoeBF16Forward(Context(x), x, topk_ids, topk_weights, weight1, weight2, max_recv_tokens, rank_,
                                       world_size_, SymmetricBufferBytes(), ffts_addr_, SymmetricBuffer(), swiglu_limit,
                                       save_gmm1_output);
    }

    ops::BackwardResult backward(torch::Tensor grad_y, torch::Tensor topk_weights, torch::Tensor weight1,
                                 torch::Tensor weight2, torch::Tensor counts, torch::Tensor saved_state,
                                 torch::Tensor dispatch_output, int64_t max_recv_tokens, double swiglu_limit,
                                 const std::string& save_gmm1_output)
    {
        TORCH_CHECK(address_ != 0, "training operator has been closed");
        return ops::MegaMoeBF16Backward(Context(grad_y), grad_y, topk_weights, weight1, weight2, counts, saved_state,
                                        dispatch_output, max_recv_tokens, rank_, world_size_, SymmetricBufferBytes(),
                                        ffts_addr_, SymmetricBuffer(), swiglu_limit, save_gmm1_output);
    }

private:
    torch::Tensor Context(const torch::Tensor& like)
    {
        TORCH_CHECK(like.defined(), "MegaMoe BF16 context reference tensor must be defined");
        TORCH_CHECK(like.device().type() == c10::DeviceType::PrivateUse1,
                    "MegaMoe BF16 context reference tensor must be an NPU tensor");
        TORCH_CHECK(like.is_contiguous(), "MegaMoe BF16 context reference tensor must be contiguous");
        TORCH_CHECK(like.scalar_type() == torch::kBFloat16, "MegaMoe BF16 context reference tensor must be bfloat16");
        const int rank = rank_;
        const int world_size = world_size_;
        if (!context_.defined()) {
            auto new_context = torch::empty({kMegaMoeContextBytes}, like.options().dtype(torch::kUInt8));
            InitializeMegaMoeBF16Context(new_context, SymmetricBuffer(), rank, world_size);
            // Publish the persistent state only after initialization succeeds.
            // A failed first call must remain retryable instead of leaving a
            // defined but partially initialized context behind.
            context_ = new_context;
            context_rank_ = rank;
            context_world_size_ = world_size;
        } else {
            TORCH_CHECK(context_.device() == like.device(), "MegaMoe BF16 context device changed across calls");
            TORCH_CHECK(context_rank_ == rank && context_world_size_ == world_size,
                        "MegaMoe BF16 SHMEM rank/world size changed across calls");
        }
        return context_;
    }

    void* SymmetricBuffer() const
    {
        return reinterpret_cast<void*>(address_);
    }
    std::size_t SymmetricBufferBytes() const
    {
        return static_cast<std::size_t>(bytes_);
    }
    uintptr_t address_;
    int64_t bytes_;
    int rank_;
    int world_size_;
    uint64_t ffts_addr_;
    torch::Tensor context_;
    int context_rank_ = -1;
    int context_world_size_ = -1;
};

}  // namespace deepep::megamoe_train

namespace py = pybind11;
PYBIND11_MODULE(_C, module)
{
    using deepep::megamoe_train::BF16Executor;
    namespace ops = deepep::megamoe_train::ops;
    py::class_<ops::ForwardResult>(module, "ForwardResult")
        .def_readonly("y", &ops::ForwardResult::y)
        .def_readonly("counts", &ops::ForwardResult::counts)
        .def_readonly("saved_state", &ops::ForwardResult::saved_state)
        .def_readonly("dispatch_output", &ops::ForwardResult::dispatch_output);
    py::class_<ops::BackwardResult>(module, "BackwardResult")
        .def_readonly("dx", &ops::BackwardResult::dx)
        .def_readonly("dtopk_weights", &ops::BackwardResult::dtopk_weights)
        .def_readonly("dw1", &ops::BackwardResult::dw1)
        .def_readonly("dw2", &ops::BackwardResult::dw2);
    py::class_<BF16Executor>(module, "BF16")
        .def(py::init<uintptr_t, int64_t, int, int, torch::Tensor>())
        .def("forward", &BF16Executor::forward)
        .def("backward", &BF16Executor::backward)
        .def("close", &BF16Executor::close);
}
