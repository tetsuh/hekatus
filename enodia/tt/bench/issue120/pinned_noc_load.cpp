// SPDX-License-Identifier: Apache-2.0
// Fixed NoC reads and writes to a pinned host allocation (Q3).
#include <cstdint>

#include "api/dataflow/dataflow_api.h"

namespace {
constexpr uint32_t cb_scratch = 0;
constexpr uint32_t page_bytes = 4096;
constexpr uint32_t read_cmd_buf = 0;
constexpr uint32_t write_cmd_buf = 1;
constexpr uint32_t clock_gap_limit = 0x7fffffff;
constexpr uint8_t noc_mode = DM_DEDICATED_NOC;
constexpr CQNocFlags read_flags = static_cast<CQNocFlags>(
    CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN | CQ_NOC_FLAG_NOC);
constexpr CQNocFlags write_flags = static_cast<CQNocFlags>(
    CQ_NOC_FLAG_SRC | CQ_NOC_FLAG_DST | CQ_NOC_FLAG_LEN | CQ_NOC_FLAG_NOC);

struct WrapTrackedClock {
    uint32_t previous = 0;
    uint64_t extended = 0;
    uint32_t ambiguous = 0;

    void initialize() {
        previous = get_timestamp_32b();
        extended = previous;
    }

    uint64_t sample() {
        const uint32_t low = get_timestamp_32b();
        const uint32_t delta = low - previous;
        if (delta > clock_gap_limit) {
            ambiguous = 1;
        }
        extended += delta;
        previous = low;
        return extended;
    }

    void poll_command_buffer(uint32_t command_buffer) {
        while (!noc_cmd_buf_ready(noc_index, command_buffer)) {
            sample();
        }
    }

    void poll_reads() {
        while (!ncrisc_noc_reads_flushed(noc_index)) {
            sample();
        }
        invalidate_l1_cache();
    }

    void poll_nonposted_writes() {
        while (!ncrisc_noc_nonposted_writes_flushed(noc_index)) {
            sample();
        }
    }
};
}  // namespace

void kernel_main() {
    const uint32_t pcie_xy_enc = get_arg_val<uint32_t>(0);
    const uint64_t host_addr = static_cast<uint64_t>(get_arg_val<uint32_t>(1)) |
        (static_cast<uint64_t>(get_arg_val<uint32_t>(2)) << 32);
    const uint32_t repeat_count = get_arg_val<uint32_t>(3);

    cb_reserve_back(cb_scratch, 2);
    const uint32_t base = get_write_ptr(cb_scratch);
    auto* payload = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base);
    auto* stats = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(base + page_bytes);
    noc_read_init_state<read_cmd_buf>(noc_index);
    noc_write_init_state<write_cmd_buf>(noc_index, NOC_UNICAST_WRITE_VC);
    WrapTrackedClock clock;
    clock.initialize();
    const uint64_t load_start = clock.extended;
    uint32_t checksum = 0;
    for (uint32_t repetition = 0; repetition < repeat_count; ++repetition) {
        clock.poll_command_buffer(read_cmd_buf);
        noc_read_with_state<
            noc_mode,
            read_cmd_buf,
            read_flags,
            CQ_NOC_SEND,
            CQ_NOC_wait>(
            noc_index,
            pcie_xy_enc,
            host_addr,
            base,
            page_bytes);
        clock.poll_reads();
        checksum ^= payload[repetition % (page_bytes / sizeof(uint32_t))] ^ repetition;
        payload[0] = checksum;
        clock.poll_command_buffer(write_cmd_buf);
        noc_wwrite_with_state<
            noc_mode,
            write_cmd_buf,
            write_flags,
            CQ_NOC_SEND,
            CQ_NOC_wait,
            true,
            false>(
            noc_index,
            base,
            pcie_xy_enc,
            host_addr + page_bytes,
            page_bytes,
            1);
        clock.poll_nonposted_writes();
        clock.sample();
    }
    const uint64_t load_end = clock.sample();
    stats[0] = static_cast<uint32_t>(load_start);
    stats[1] = static_cast<uint32_t>(load_start >> 32);
    stats[2] = static_cast<uint32_t>(load_end);
    stats[3] = static_cast<uint32_t>(load_end >> 32);
    stats[4] = repeat_count;
    stats[5] = repeat_count;
    stats[6] = page_bytes;
    stats[7] = checksum;
    stats[8] = clock.ambiguous;

    clock.poll_command_buffer(write_cmd_buf);
    noc_wwrite_with_state<
        noc_mode,
        write_cmd_buf,
        write_flags,
        CQ_NOC_SEND,
        CQ_NOC_wait,
        true,
        false>(
        noc_index,
        base + page_bytes,
        pcie_xy_enc,
        host_addr + 2 * page_bytes,
        9 * sizeof(uint32_t),
        1);
    clock.poll_nonposted_writes();
    cb_push_back(cb_scratch, 2);
    cb_wait_front(cb_scratch, 2);
    cb_pop_front(cb_scratch, 2);
}
