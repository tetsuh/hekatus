"""Execute the production host seam; no board, native runner or source grep."""

import subprocess


def test_peer_runtime_indices_and_translation_are_shared_by_host_seam(tmp_path):
    source = tmp_path / "peer.cpp"
    source.write_text(r'''
#include "q4_host_sequence.hpp"
#include <cassert>
#include <stdexcept>
#include <vector>
int main() {
    using namespace issue120_q4_host;
    using namespace issue120_q4_args;
    static_assert(Producer::peer_x == 7 && Producer::peer_y == 8 && Producer::count == 9);
    static_assert(Consumer::mailbox == 11 && Consumer::peer_x == 12 && Consumer::peer_y == 13);
    static_assert(Consumer::count == 14);
    std::vector<uint32_t> producer{10,11,12,13,14,15,16};
    std::vector<uint32_t> consumer{20,21,22,23,24,25,26,27,28,29,30,999};
    const auto old_p = producer, old_c = consumer;
    unsigned resolves = 0;
    auto resolver = [&] {
        ++resolves;
        return resolve_peer_coordinates(CoreXY{0,0}, CoreXY{1,0}, [](CoreXY core) {
            return CoreXY{core.x + 1, core.y + 2};
        });
    };
    const auto passed = append_peer_runtime_args(producer, consumer, resolver);
    assert(resolves == 1);
    assert(producer.size() == Producer::count && consumer.size() == Consumer::count);
    for (unsigned i=0;i<old_p.size();++i) assert(producer[i] == old_p[i]);
    for (unsigned i=0;i<old_c.size();++i) assert(consumer[i] == old_c[i]);
    assert(producer[Producer::peer_x] == 2 && producer[Producer::peer_y] == 2);
    assert(consumer[Consumer::peer_x] == 1 && consumer[Consumer::peer_y] == 2);
    assert(consumer[Consumer::mailbox] == 999);
    assert(passed.producer.x == 1 && passed.producer.y == 2);
    assert(passed.consumer.x == 2 && passed.consumer.y == 2);
    const auto diagnostic = resolver();
    assert(diagnostic.producer.x == passed.producer.x && diagnostic.consumer.y == passed.consumer.y);
    assert(pack_coordinate(diagnostic.producer) == 0x102);
    assert(pack_coordinate(diagnostic.consumer) == 0x202);
    assert(coordinate_classification(0x102, CoreXY{1,2}, CoreXY{1,2})[0] == 'A');
    assert(coordinate_classification(0x102, CoreXY{1,5}, CoreXY{1,2})[0] == 'B');
    assert(coordinate_classification(0x105, CoreXY{1,5}, CoreXY{1,2})[0] == 'C');
    bool rejected = false;
    try { append_peer_runtime_args(producer, consumer, resolver); }
    catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
''')
    binary = tmp_path / "peer"
    subprocess.run(["g++", "-std=c++20", "-I", "enodia/tt/bench/issue120", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True, timeout=10)
