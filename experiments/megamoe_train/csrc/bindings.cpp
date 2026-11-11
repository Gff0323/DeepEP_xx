// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
// Modified by xixing 2026

#include "bf16/kernel_launch.hpp"
#include "bf16/training.hpp"

#include "kernels/bf16/common/ascend_timer_v2.hpp"

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
    // Device timer surface: the buffer size is a compile-time layout constant,
    // and support is answered by the linked device SO, not by Python.
    module.def("timer_buffer_size", []() { return AscendTimer::TOTAL_BUFFER_SIZE; },
               "int64 elements required by one MegaMoe BF16 training timer buffer");
    module.def("timer_enabled", []() { return deepep_train_mega_moe_bf16_timer_enabled() != 0; },
               "whether the loaded device SO was built with DEEPEP_MEGAMOE_TRAIN_TIMER=ON");
}
