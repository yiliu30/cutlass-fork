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
                  int tiles, int spans_per_tile, int qk_spans, double us_per_tick) {
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
  std::ofstream out(path);
  out.imbue(std::locale::classic());
  out << std::fixed << std::setprecision(6)
      << "{\"displayTimeUnit\":\"ns\",\"otherData\":{\"clock\":\"subgroup-relative, not synchronized\","
      << "\"microsecondsPerTick\":" << us_per_tick << ",\"workgroup\":0},\"traceEvents\":[";
  out << "\n{\"ph\":\"M\",\"name\":\"process_name\",\"pid\":1,\"tid\":1,"
         "\"args\":{\"name\":\"BF16 FMHA prefill\"}}";
  for (int sg = 0; sg < cutlass::fmha::fmha_trace_subgroups; ++sg) {
    int tid = sg + 2;
    out << ",\n{\"ph\":\"M\",\"name\":\"thread_name\",\"pid\":1,\"tid\":" << tid
        << ",\"args\":{\"name\":\"WG 0 / SG " << sg << " (relative)\"}}";
    int base = sg * tiles * spans_per_tile;
    auto origin = spans[base].begin;
    for (int tile = 0; tile < tiles; ++tile) {
      for (int i = 0; i < spans_per_tile; ++i) {
        auto [begin, end] = spans[base + tile * spans_per_tile + i];
        if (begin < origin) {
          std::cerr << "FMHA subgroup clock moved backwards.\n";
          return false;
        }
        char const* phase = i < qk_spans ? "QK DPAS" : i == qk_spans ? "Online softmax" : "PV DPAS";
        out << ",\n{\"ph\":\"X\",\"cat\":\"FMHA\",\"name\":\"" << phase
            << "\",\"pid\":1,\"tid\":" << tid << ",\"ts\":" << (begin - origin) * us_per_tick
            << ",\"dur\":" << (end - begin) * us_per_tick
            << ",\"args\":{\"kv_tile\":" << tile << ",\"subtile\":"
            << (i <= qk_spans ? (i == qk_spans ? 0 : i) : i - qk_spans - 1) << "}}";
      }
    }
  }
  out << "\n]}\n";
  out.close();
  if (!out) {
    std::cerr << "Could not write FMHA trace to " << path << '\n';
    return false;
  }
  std::cout << "Wrote " << spans.size() << " FMHA spans to " << path << '\n';
  return true;
}

}  // namespace fmha_trace
