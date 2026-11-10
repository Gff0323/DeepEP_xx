// Copyright (c) 2026, Lu Lu
// Modified by guofanfeng 2026
//
// Every BF16 timer reference stays behind one compile-time switch.  Without
// -DDEEPEP_BF16_TIMER=1 the timer objects, the cycle reads, the pipe barriers
// and the GM writes are all removed by preprocessing, so a release build of
// the operator is byte-for-byte the same operator as before.
//
// Ported from ascend_deepep_hkq (7eefe6e^), where the same three macros
// threaded an optional AscendTimerDevice through the GMM helpers.

#ifndef DEEPEP_MEGAMOE_TRAIN_BF16_TIMER_H
#define DEEPEP_MEGAMOE_TRAIN_BF16_TIMER_H

#if defined(DEEPEP_BF16_TIMER)
#include "ascend_timer_v2_device.hpp"
#define MEGA_MOE_BF16_TIMER_ENABLED 1
#define MEGA_MOE_BF16_TIMER_PARAM(name) , AscendTimerDevice& name
#define MEGA_MOE_BF16_TIMER_ARG(name) , name
#define MEGA_MOE_BF16_TIMER_BLOCK(code) TIMER_BLOCK(code)
#else
#define MEGA_MOE_BF16_TIMER_ENABLED 0
#define MEGA_MOE_BF16_TIMER_PARAM(name)
#define MEGA_MOE_BF16_TIMER_ARG(name)
#define MEGA_MOE_BF16_TIMER_BLOCK(code) \
    do {                               \
    } while (0)
#endif

#endif  // DEEPEP_MEGAMOE_TRAIN_BF16_TIMER_H
