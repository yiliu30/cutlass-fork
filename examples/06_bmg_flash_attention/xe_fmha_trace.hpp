#pragma once

#include <algorithm>
#include <array>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <locale>
#include <string>
#include <utility>
#include <vector>

namespace fmha_trace {

inline bool supports_placement(sycl::device const& device) {
  namespace ext = sycl::ext::oneapi::experimental;
  auto arch = device.get_info<ext::info::device::architecture>();
  return arch == ext::architecture::intel_gpu_bmg_g21 || arch == ext::architecture::intel_gpu_bmg_g31;
}

inline void json_string(std::ostream& out, std::string const& value) {
  constexpr char hex[] = "0123456789abcdef";
  out << '"';
  for (unsigned char c : value) {
    if (c == '"' || c == '\\') out << '\\' << char(c);
    else if (c < 0x20) out << "\\u00" << hex[c >> 4] << hex[c & 15];
    else out << char(c);
  }
  out << '"';
}

struct Xe2Placement {
  unsigned slice;
  unsigned subslice;
  unsigned eu;
  unsigned thread;

  static Xe2Placement decode(uint32_t sr0) {
    // Xe2 SR0.0 layout used by IGC GenXSubtarget::getHWThreadIdReservedBits.
    return {(sr0 >> 11) & 31, (sr0 >> 8) & 3, (sr0 >> 4) & 7, sr0 & 7};
  }
};

inline double microseconds_per_tick(sycl::queue const& queue) {
  sycl::queue timer(queue.get_context(), queue.get_device(),
                    sycl::property::queue::enable_profiling{});
  cutlass::device_memory::allocation<uint64_t> observed(1);
  auto* ticks = observed.get();
  auto sample = [&](uint64_t target) {
    auto event = timer.parallel_for(sycl::nd_range<1>{16, 16},
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          auto start = cutlass::fmha::fmha_trace_clock();
          uint64_t elapsed;
          do {
            elapsed = cutlass::fmha::fmha_trace_clock() - start;
          } while (elapsed < target);
          if (item.get_local_linear_id() == 0) {
            *ticks = elapsed;
          }
        });
    event.wait_and_throw();
    uint64_t elapsed;
    timer.memcpy(&elapsed, ticks, sizeof(elapsed)).wait_and_throw();
    using namespace sycl::info::event_profiling;
    auto ns = event.get_profiling_info<command_end>() - event.get_profiling_info<command_start>();
    return std::pair{double(elapsed), double(ns)};
  };

  // Subtract the short dispatch to remove most fixed measurement overhead.
  sample(1000);
  std::array<double, 3> estimates{};
  for (auto& estimate : estimates) {
    auto [short_ticks, short_ns] = sample(50000);
    auto [long_ticks, long_ns] = sample(200000);
    if (long_ticks <= short_ticks || long_ns <= short_ns) {
      return 0;
    }
    estimate = (long_ns - short_ns) / (long_ticks - short_ticks) / 1000;
  }
  std::sort(estimates.begin(), estimates.end());
  return estimates[1];
}

inline bool write(std::string const& path, std::vector<cutlass::fmha::FmhaTraceSpan> const& spans,
                  std::vector<uint32_t> const& placements, sycl::device const& device,
                  int subgroups, int tiles, int spans_per_tile, int qk_spans, double us_per_tick) {
  using namespace cutlass::fmha;
  if (us_per_tick <= 0) {
    std::cerr << "Could not calibrate the FMHA subgroup clock.\n";
    return false;
  }
  for (size_t i = 0; i < spans.size(); ++i) {
    if (spans[i].begin == 0 || spans[i].end < spans[i].begin) {
      std::cerr << "FMHA trace span " << i << " is missing or invalid.\n";
      return false;
    }
  }
  if (std::find(placements.begin(), placements.end(), fmha_trace_no_placement) != placements.end()) {
    std::cerr << "FMHA trace is missing a hardware placement sample.\n";
    return false;
  }
  namespace intel_info = sycl::ext::intel::info::device;
  auto name = device.get_info<sycl::info::device::name>();
  auto pci = device.has(sycl::aspect::ext_intel_pci_address)
      ? device.get_info<intel_info::pci_address>() : std::string{};
  std::ofstream out(path);
  out.imbue(std::locale::classic());
  out << std::fixed << std::setprecision(6)
      << "{\"displayTimeUnit\":\"ns\",\"otherData\":{\"clock\":\"subgroup-relative, not synchronized\","
      << "\"microsecondsPerTick\":" << us_per_tick << ",\"workgroup\":0,\"subgroups\":" << subgroups
      << ",\"kvTiles\":" << tiles << ",\"device\":{\"name\":";
  json_string(out, name);
  out << ",\"driver_version\":";
  json_string(out, device.get_info<sycl::info::device::driver_version>());
  if (!pci.empty()) {
    out << ",\"pci_address\":";
    json_string(out, pci);
  }
  if (device.has(sycl::aspect::ext_intel_gpu_slices)) {
    out << ",\"slices\":" << device.get_info<intel_info::gpu_slices>();
  }
  if (device.has(sycl::aspect::ext_intel_gpu_subslices_per_slice)) {
    out << ",\"subslices_per_slice\":" << device.get_info<intel_info::gpu_subslices_per_slice>();
  }
  if (device.has(sycl::aspect::ext_intel_gpu_eu_count)) {
    out << ",\"eus\":" << device.get_info<intel_info::gpu_eu_count>();
  }
  if (device.has(sycl::aspect::ext_intel_gpu_hw_threads_per_eu)) {
    out << ",\"hardware_threads_per_eu\":" << device.get_info<intel_info::gpu_hw_threads_per_eu>();
  }
  out << "},\"placement_sampling\":";
  json_string(out, placements.empty() ? "unavailable on this device"
                                      : "Xe2 SR0.0 sampled once per subgroup at KV tile entry");
  out << "},\"traceEvents\":[\n{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":1,\"tid\":1,\"args\":{\"name\":";
  json_string(out, "BF16 FMHA prefill | " + name + (pci.empty() ? "" : " @ " + pci));
  out << "}}";
  constexpr std::array<char const*, fmha_trace_qk_phase_count> qk_labels{
    "K load", "K reorder", "QK GEMM"
  };
  constexpr std::array<char const*, fmha_trace_softmax_phase_count> softmax_labels{
    "Online softmax", "Row max", "Rescale factor", "Softmax exp + partial sum", "P pack / reorder"
  };
  constexpr std::array<char const*, fmha_trace_pv_phase_count> pv_labels{
    "V load", "V reorder", "Out Acc Rescale", "PV GEMM"
  };
  for (int sg = 0; sg < subgroups; ++sg) {
    int tid = sg + 2;
    out << ",\n{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":" << tid
        << ",\"args\":{\"name\":\"WG 0 / SG " << sg << " (relative)";
    if (!placements.empty()) {
      auto hw = Xe2Placement::decode(placements[sg * tiles]);
      out << " | tile 0: S" << hw.slice << " SS" << hw.subslice << " EU" << hw.eu << " T" << hw.thread;
    }
    out << "\"}}";
    int base = sg * tiles * spans_per_tile;
    auto origin = spans[base + FmhaTraceLayout::tile_start].begin;
    for (int tile = 0; tile < tiles; ++tile) {
      auto sr0 = placements.empty() ? fmha_trace_no_placement : placements[sg * tiles + tile];
      for (int i = 0; i < spans_per_tile; ++i) {
        auto [begin, end] = spans[base + tile * spans_per_tile + i];
        if (begin < origin) {
          std::cerr << "FMHA subgroup clock moved backwards.\n";
          return false;
        }
        if (i == FmhaTraceLayout::tile_start) {
          out << ",\n{\"ph\":\"i\",\"s\":\"t\",\"cat\":\"FMHA\",\"name\":\"KV tile " << tile
              << " start\",\"pid\":1,\"tid\":" << tid << ",\"ts\":" << (begin - origin) * us_per_tick
              << ",\"args\":{\"workgroup_id\":0,\"subgroup_id\":" << sg
              << ",\"kv_tile\":" << tile << ",\"timestamp_source\":\"device tile entry\"}}";
          continue;
        }
        char const* phase;
        int subtile = 0;
        if (i == FmhaTraceLayout::v_prefetch) {
          phase = "V prefetch issue";
        } else if (i < FmhaTraceLayout::k_prefetch(qk_spans)) {
          int qk_index = i - FmhaTraceLayout::qk_begin;
          phase = qk_labels[qk_index % fmha_trace_qk_phase_count];
          subtile = qk_index / fmha_trace_qk_phase_count;
        } else if (i == FmhaTraceLayout::k_prefetch(qk_spans)) {
          phase = "K prefetch issue";
        } else if (i < FmhaTraceLayout::pv_begin(qk_spans)) {
          phase = softmax_labels[i - FmhaTraceLayout::softmax(qk_spans)];
        } else {
          int pv_index = i - FmhaTraceLayout::pv_begin(qk_spans);
          phase = pv_labels[pv_index % fmha_trace_pv_phase_count];
          subtile = pv_index / fmha_trace_pv_phase_count;
        }
        out << ",\n{\"ph\":\"X\",\"cat\":\"FMHA\",\"name\":\"" << phase
            << "\",\"pid\":1,\"tid\":" << tid << ",\"ts\":" << (begin - origin) * us_per_tick
            << ",\"dur\":" << (end - begin) * us_per_tick
            << ",\"args\":{\"workgroup_id\":0,\"subgroup_id\":" << sg
            << ",\"kv_tile\":" << tile << ",\"subtile\":" << subtile;
        if (i >= FmhaTraceLayout::pv_begin(qk_spans) &&
            (i - FmhaTraceLayout::pv_begin(qk_spans)) % fmha_trace_pv_phase_count == fmha_trace_pv_dpas) {
          out << ",\"updates\":\"output_accumulator\"";
        }
        if (sr0 != fmha_trace_no_placement) {
          auto hw = Xe2Placement::decode(sr0);
          out << ",\"slice_id\":" << hw.slice << ",\"subslice_id\":" << hw.subslice
              << ",\"eu_id\":" << hw.eu << ",\"hw_thread_id\":" << hw.thread << ",\"sr0_low16\":" << sr0;
        }
        out << "}}";
      }
    }
  }
  out << "\n]}\n";
  out.close();
  if (!out) {
    std::cerr << "Could not write FMHA trace to " << path << '\n';
    return false;
  }
  auto tile_starts = size_t(subgroups) * tiles;
  std::cout << "Wrote " << spans.size() - tile_starts << " FMHA spans and " << tile_starts
            << " KV tile start flags to " << path << '\n';
  return true;
}

}  // namespace fmha_trace
