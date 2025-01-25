#pragma once

#include "pcg_basic.h"
#include <bit>

struct RNG {
    pcg32_random_t pcg_state;

    RNG()
    {
        init(0, 0);
    }

    void init(uint64_t init_state, uint64_t stream_id)
    {
        pcg32_srandom_r(&pcg_state, init_state, stream_id);
    }

    // uniformly distributed 32-bit number
    uint32_t get_uint()
    {
        return pcg32_random_r(&pcg_state);
    }

   // [0, 1)
    float get_float()
    {
        uint32_t i = (pcg32_random_r(&pcg_state) >> 9) | 0x3f800000u;
        float f = std::bit_cast<float>(i);
        return f - 1.f;
    }

    // [-1, 1)
    float get_float_signed()
    {
        return (2.f * get_float() - 1.f);
    }
};
