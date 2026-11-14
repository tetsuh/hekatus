"""CPU-only admission and constexpr guard tests for the V3 accessor contract."""

import subprocess


def test_single_core_sharded_accessor_admission(tmp_path):
    source = tmp_path / "admission.cpp"
    source.write_text(r'''
#include "q4_host_sequence.hpp"
#include <cassert>
#include <vector>
int main() {
 using namespace issue120_q4_host;
 const CoreXY peer{2,2};
 assert(validate_single_core_accessor({1,4096,1,1,4,4,0x202},peer,4)==6);
 assert(validate_single_core_accessor({1,4096,2,1,4,1,4,1,0x202},peer,4)==8);
 for (auto bad : std::vector<std::vector<uint32_t>>{
     {0,4096}, {1,4096,1,1,4,4,0x205}, {1,4096,1,1,4,4},
     {1,4096,1,2,4,4,0x202}, {1,4096,1,1,3,4,0x202}, {65,4096,1,1,4,4,0x202}}) {
   bool rejected=false;
   try { validate_single_core_accessor(bad,peer,4); }
   catch (const std::invalid_argument&) { rejected=true; }
   assert(rejected);
 }
}
''')
    binary = tmp_path / "admission"
    subprocess.run(["g++", "-std=c++17", "-I", "enodia/tt/bench/issue120", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def test_non_sharded_snapshot_never_instantiates_a_missing_bank_word(tmp_path):
    source = tmp_path / "guard.cpp"
    source.write_text(r'''
#include <cstdint>
#include <cassert>
template<unsigned I> constexpr uint32_t fake_cta() {
 static_assert(I<4,"Index out of range"); return I==0 ? 0 : 4096;
}
#define get_compile_time_arg_val(i) fake_cta<i>()
#include "q4_accessor_diagnostics.hpp"
struct Unsharded { static constexpr bool is_sharded=false; static constexpr uint32_t BankCoordsCTAOffset=4; };
struct Sharded { static constexpr bool is_sharded=true; static constexpr uint32_t BankCoordsCTAOffset=2; };
int main() {
 static_assert(issue120_q4::accessor_bank_word<Unsharded>()==0);
 static_assert(issue120_q4::accessor_bank_slot<Unsharded>()==0xffffffffu);
 static_assert(issue120_q4::accessor_bank_word<Sharded>()==4096);
 static_assert(issue120_q4::accessor_bank_slot<Sharded>()==2);
}
''')
    binary = tmp_path / "guard"
    subprocess.run(["g++", "-std=c++17", "-I", "enodia/tt/bench/issue120", str(source), "-o", str(binary)], check=True)
    subprocess.run([str(binary)], check=True)
