// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cstdint>

namespace issue120_q4 {
// A public offset is not a valid slot for a non-sharded accessor. Keeping the
// read dependent on Args prevents instantiation of an out-of-range get_ct_arg.
template <typename Args>
constexpr uint32_t accessor_bank_word() {
    if constexpr (Args::is_sharded) {
        return get_compile_time_arg_val(Args::BankCoordsCTAOffset);
    } else {
        return 0;
    }
}

template <typename Args>
constexpr uint32_t accessor_bank_slot() {
    if constexpr (Args::is_sharded) {
        return Args::BankCoordsCTAOffset;
    } else {
        return 0xffffffffu;
    }
}
}
