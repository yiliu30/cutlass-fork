#pragma once

// Intra Kernel Tracer (IKT) for Intel Xe / SYCL.
//
// Device side: each subgroup (the Xe analogue of a CUDA warp) records named
// begin/end/mark events into its own circular buffer. Only the subgroup leader
// records, keeping the hot path small.
//
// Host side: allocate USM buffers, launch the instrumented kernel, and dump a
// compact binary trace. scripts/ikt_to_perfetto.py converts that binary to
// Perfetto/Chrome Trace JSON and a summary JSON.

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/experimental/clock.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifndef IKT_TRACE_ENABLED
#define IKT_TRACE_ENABLED 1
#endif

namespace ikt {

inline constexpr uint16_t kBegin = 0;
inline constexpr uint16_t kEnd = 1;
inline constexpr uint16_t kMark = 2;

// Depth sentinel used by the legacy flat IKT_BEGIN/IKT_END macros. Events
// written through ikt::DeviceTracer carry their real range depth (0..254).
inline constexpr uint8_t kFlatDepth = 0xFF;

// 16-byte event. The device writer stores this as one 128-bit vector store.
// The high byte of `type` is the range depth for stack-based push/pop events.
struct alignas(16) Event {
  uint64_t ts;       // subgroup-clock ticks
  uint16_t id;       // region id
  uint16_t type;     // low 8 bits: kBegin/kEnd/kMark; high 8 bits: depth
  uint16_t wg;       // workgroup linear id (low 16 bits)
  uint16_t sg_info;  // packed subgroup + Xe2 placement
};
static_assert(sizeof(Event) == 16, "Event must be 16 bytes");
static_assert(alignof(Event) == 16, "Event must be 16-byte aligned");

struct GlobalBuffer {
  Event* events = nullptr;
  uint32_t* counters = nullptr;
  uint32_t cap = 0;
  uint32_t sgs_per_wg = 0;
  uint32_t wg_base = 0;
  uint32_t wg_count = 0;
};

namespace detail {

// Xe2 SR0.0 layout used by IGC GenXSubtarget::getHWThreadIdReservedBits.
// Packed into 16 bits:
//   bits  0..4 : subgroup id (0..31)
//   bits  5..7 : slice id   (0..7)
//   bits  8..9 : subslice id (0..3)
//   bits 10..12: EU id      (0..7)
//   bits 13..15: HW thread  (0..7)
inline uint32_t read_sr0() {
#if defined(__SYCL_DEVICE_ONLY__) && defined(SYCL_INTEL_TARGET)
  uint32_t sr0;
  asm volatile("mov (M1_NM, 1) %0(0,0)<1> %%sr0(0,0)<0;1,0>" : "=rw"(sr0));
  return sr0 & 0xffffu;
#else
  return 0;
#endif
}

inline uint16_t pack_sg_info(uint32_t sg_id) {
  const uint32_t sr0 = read_sr0();
  const uint32_t slice = (sr0 >> 11) & 0x7u;
  const uint32_t subslice = (sr0 >> 8) & 0x3u;
  const uint32_t eu = (sr0 >> 4) & 0x7u;
  const uint32_t thread = sr0 & 0x7u;
  return uint16_t((sg_id & 0x1fu) | (slice << 5) | (subslice << 8) |
                  (eu << 10) | (thread << 13));
}

inline uint64_t clock_ticks() {
  namespace timing = sycl::ext::oneapi::experimental;
  return timing::clock<timing::clock_scope::sub_group>();
}

}  // namespace detail

// Per-subgroup circular buffer recorder. CAP must be a power of two.
// If a subgroup records more than CAP events, only the last CAP are retained.
template <uint32_t Cap>
struct SubgroupContext {
  static_assert((Cap & (Cap - 1u)) == 0u, "Cap must be a power of two");
  static constexpr uint32_t capacity = Cap;

  uint32_t base = 0;
  uint32_t cnt = 0;
  uint32_t slot = 0;
  uint32_t wg = 0;
  uint16_t sg_info = 0;
  bool active = false;

  inline void init(GlobalBuffer const& buf, uint32_t workgroup, uint32_t sg_id) {
    active = (buf.events != nullptr) && (buf.counters != nullptr) &&
             (buf.cap == Cap) && (sg_id < buf.sgs_per_wg) &&
             (workgroup >= buf.wg_base) &&
             (workgroup < buf.wg_base + buf.wg_count);
    if (!active) {
      return;
    }
    wg = workgroup & 0xffffu;
    sg_info = detail::pack_sg_info(sg_id);
    slot = (workgroup - buf.wg_base) * buf.sgs_per_wg + sg_id;
    base = slot * Cap;
    cnt = 0;
  }

  inline void record(GlobalBuffer const& buf, uint16_t id, uint16_t type,
                     uint8_t range_depth) {
    if (!active) {
      return;
    }
    const uint64_t ts = detail::clock_ticks();
    const uint32_t idx = base + (cnt & (Cap - 1u));
    const uint32_t w0 = uint32_t(ts);
    const uint32_t w1 = uint32_t(ts >> 32);
    const uint32_t type_depth = uint32_t(type) | (uint32_t(range_depth) << 8);
    const uint32_t w2 = uint32_t(id) | (type_depth << 16);
    const uint32_t w3 = wg | (uint32_t(sg_info) << 16);
    auto* dst = reinterpret_cast<sycl::vec<uint32_t, 4>*>(buf.events + idx);
    *dst = sycl::vec<uint32_t, 4>{w0, w1, w2, w3};
    ++cnt;
  }

  // Legacy flat begin/end/mark: no nesting information is recorded.
  inline void record_flat(GlobalBuffer const& buf, uint16_t id, uint16_t type) {
    record(buf, id, type, kFlatDepth);
  }

  inline void flush(GlobalBuffer const& buf) {
    if (active) {
      buf.counters[slot] = cnt;
    }
  }
};

template <uint32_t Cap, uint32_t MaxDepth = 16>
class ScopedRange;

// Device-side facade with the same push/pop/mark shape as IKET. The range
// stack lives here, not in SubgroupContext, so legacy flat-macro kernels keep
// the original lightweight context.
template <uint32_t Cap, uint32_t MaxDepth = 16>
class DeviceTracer {
 public:
  using Context = SubgroupContext<Cap>;
  static constexpr uint32_t max_depth = MaxDepth;

  DeviceTracer(Context& ctx, GlobalBuffer const& buf, sycl::sub_group sg)
      : ctx_(ctx), buf_(buf), leader_(sg.leader()) {}

  void range_push(uint16_t id) {
    if constexpr (IKT_TRACE_ENABLED) {
      if (!leader_) {
        return;
      }
      if (depth_ < MaxDepth) {
        stack_[depth_] = id;
      }
      ctx_.record(buf_, id, kBegin, depth_);
      if (depth_ < 0xFFu) {
        ++depth_;
      }
    }
  }

  void range_pop() {
    if constexpr (IKT_TRACE_ENABLED) {
      if (!leader_ || depth_ == 0) {
        return;
      }
      --depth_;
      const uint16_t id = (depth_ < MaxDepth) ? stack_[depth_] : 0;
      ctx_.record(buf_, id, kEnd, depth_);
    }
  }

  void mark(uint16_t id) {
    if constexpr (IKT_TRACE_ENABLED) {
      if (leader_) {
        ctx_.record(buf_, id, kMark, depth_);
      }
    }
  }

  ScopedRange<Cap, MaxDepth> scoped(uint16_t id) {
    return ScopedRange<Cap, MaxDepth>(*this, id);
  }

 private:
  Context& ctx_;
  GlobalBuffer buf_;
  bool leader_;
  uint8_t depth_ = 0;
  uint16_t stack_[MaxDepth] = {};
};

template <uint32_t Cap, uint32_t MaxDepth>
class ScopedRange {
 public:
  ScopedRange(DeviceTracer<Cap, MaxDepth>& tracer, uint16_t id) : tracer_(&tracer) {
    tracer_->range_push(id);
  }
  ~ScopedRange() {
    if (tracer_) {
      tracer_->range_pop();
    }
  }
  ScopedRange(ScopedRange const&) = delete;
  ScopedRange& operator=(ScopedRange const&) = delete;
  ScopedRange(ScopedRange&& other) noexcept : tracer_(other.tracer_) {
    other.tracer_ = nullptr;
  }
  ScopedRange& operator=(ScopedRange&&) = delete;

 private:
  DeviceTracer<Cap, MaxDepth>* tracer_;
};

template <uint32_t Cap>
DeviceTracer(SubgroupContext<Cap>&, GlobalBuffer const&, sycl::sub_group)
    -> DeviceTracer<Cap>;

template <uint32_t Cap>
DeviceTracer<Cap> make_device_tracer(SubgroupContext<Cap>& ctx,
                                     GlobalBuffer const& buf,
                                     sycl::sub_group sg) {
  return DeviceTracer<Cap>(ctx, buf, sg);
}

#if IKT_TRACE_ENABLED
#define IKT_CTX_TYPE(cap) ::ikt::SubgroupContext<(cap)>
#define IKT_CTX_INIT(ctx, gbuf, item, sg)                                      \
  do {                                                                             \
    if ((sg).leader()) {                                                           \
      (ctx).init((gbuf), uint32_t((item).get_group_linear_id()),                    \
                 uint32_t((sg).get_group_linear_id()));                             \
    }                                                                              \
  } while (0)
#define IKT_BEGIN(ctx, gbuf, sg, id)                                            \
  do {                                                                             \
    if ((sg).leader()) {                                                           \
      (ctx).record_flat((gbuf), uint16_t(id), ::ikt::kBegin);                    \
    }                                                                              \
  } while (0)
#define IKT_END(ctx, gbuf, sg, id)                                              \
  do {                                                                             \
    if ((sg).leader()) {                                                           \
      (ctx).record_flat((gbuf), uint16_t(id), ::ikt::kEnd);                      \
    }                                                                              \
  } while (0)
#define IKT_MARK(ctx, gbuf, sg, id)                                             \
  do {                                                                             \
    if ((sg).leader()) {                                                           \
      (ctx).record_flat((gbuf), uint16_t(id), ::ikt::kMark);                     \
    }                                                                              \
  } while (0)
#define IKT_BEGIN_IF(ctx, gbuf, sg, id, cond)                                    \
  do {                                                                             \
    if ((sg).leader() && (cond)) {                                                 \
      (ctx).record_flat((gbuf), uint16_t(id), ::ikt::kBegin);                    \
    }                                                                              \
  } while (0)
#define IKT_END_IF(ctx, gbuf, sg, id, cond)                                      \
  do {                                                                             \
    if ((sg).leader() && (cond)) {                                                 \
      (ctx).record_flat((gbuf), uint16_t(id), ::ikt::kEnd);                      \
    }                                                                              \
  } while (0)
#define IKT_FLUSH(ctx, gbuf)                                                    \
  do {                                                                             \
    if ((sg).leader()) {                                                           \
      (ctx).flush((gbuf));                                                         \
    }                                                                              \
  } while (0)
#else
#define IKT_CTX_TYPE(cap) ::ikt::SubgroupContext<(cap)>
#define IKT_CTX_INIT(ctx, gbuf, item, sg) do { (void)(ctx); } while (0)
#define IKT_BEGIN(ctx, gbuf, sg, id) do { (void)(ctx); } while (0)
#define IKT_END(ctx, gbuf, sg, id) do { (void)(ctx); } while (0)
#define IKT_MARK(ctx, gbuf, sg, id) do { (void)(ctx); } while (0)
#define IKT_BEGIN_IF(ctx, gbuf, sg, id, cond) do { (void)(ctx); } while (0)
#define IKT_END_IF(ctx, gbuf, sg, id, cond) do { (void)(ctx); } while (0)
#define IKT_FLUSH(ctx, gbuf) do { (void)(ctx); } while (0)
#endif

#define IKT_RANGE_PUSH(tracer, id) do { (tracer).range_push(uint16_t(id)); } while (0)
#define IKT_RANGE_POP(tracer) do { (tracer).range_pop(); } while (0)
#define IKT_TRACER_MARK(tracer, id) do { (tracer).mark(uint16_t(id)); } while (0)
#define IKT_SCOPE_IMPL(tracer, id, ctr) auto ikt_scope_##ctr = (tracer).scoped(uint16_t(id))
#define IKT_SCOPE(tracer, id) IKT_SCOPE_IMPL(tracer, id, __COUNTER__)

namespace detail {

inline double calibrate_ns_per_tick(sycl::queue const& base_queue) {
  sycl::queue timer(base_queue.get_context(), base_queue.get_device(),
                    sycl::property::queue::enable_profiling{});
  auto* ticks = sycl::malloc_device<uint64_t>(1, timer);
  if (!ticks) {
    throw std::runtime_error("ikt: failed to allocate calibration buffer");
  }

  auto sample = [&](uint64_t target) {
    auto event = timer.parallel_for(
        sycl::nd_range<1>{16, 16}, [=](sycl::nd_item<1> item)
            [[sycl::reqd_sub_group_size(16)]] {
              const uint64_t start = detail::clock_ticks();
              uint64_t elapsed = 0;
              do {
                elapsed = detail::clock_ticks() - start;
              } while (elapsed < target);
              if (item.get_local_linear_id() == 0) {
                *ticks = elapsed;
              }
            });
    event.wait_and_throw();
    uint64_t elapsed = 0;
    timer.memcpy(&elapsed, ticks, sizeof(elapsed)).wait_and_throw();
    using namespace sycl::info::event_profiling;
    const uint64_t ns =
        event.get_profiling_info<command_end>() -
        event.get_profiling_info<command_start>();
    return std::pair<double, double>{double(elapsed), double(ns)};
  };

  // Warm the GPU and let the clock governor settle before sampling.
  sample(5000000);
  std::array<double, 5> estimates{};
  for (auto& estimate : estimates) {
    auto [short_ticks, short_ns] = sample(5000000);
    auto [long_ticks, long_ns] = sample(20000000);
    if (long_ticks <= short_ticks || long_ns <= short_ns) {
      sycl::free(ticks, timer);
      return 0.0;
    }
    estimate = (long_ns - short_ns) / (long_ticks - short_ticks);
  }
  std::sort(estimates.begin(), estimates.end());
  sycl::free(ticks, timer);
  return estimates[2];
}

}  // namespace detail

// Host-side session. The queue must outlive the session.
class HostSession {
 public:
  explicit HostSession(sycl::queue& queue) : queue_(queue) {}

  HostSession(HostSession const&) = delete;
  HostSession& operator=(HostSession const&) = delete;

  ~HostSession() { destroy(); }

  void init(uint32_t cap, uint32_t grid_size, uint32_t sgs_per_wg,
            uint32_t wg_base = 0, uint32_t wg_count = 0xffffffffu) {
    destroy();
    if (cap == 0 || (cap & (cap - 1u)) != 0u) {
      throw std::invalid_argument("ikt: cap must be a power of two and > 0");
    }
    if (grid_size == 0 || sgs_per_wg == 0 || wg_base >= grid_size) {
      throw std::invalid_argument("ikt: invalid grid_size, wg_base, or sgs_per_wg");
    }
    wg_count = std::min(wg_count, grid_size - wg_base);
    if (!queue_.get_device().has(sycl::aspect::ext_oneapi_clock_sub_group)) {
      throw std::runtime_error(
          "ikt: device does not support ext_oneapi_clock_sub_group");
    }
    cap_ = cap;
    grid_ = grid_size;
    sgs_per_wg_ = sgs_per_wg;
    wg_base_ = wg_base;
    wg_count_ = wg_count;
    const size_t slots = size_t(wg_count_) * size_t(sgs_per_wg_);
    const size_t events = slots * size_t(cap_);
    d_events_ = sycl::malloc_device<Event>(events, queue_);
    d_counters_ = sycl::malloc_device<uint32_t>(slots, queue_);
    if (!d_events_ || !d_counters_) {
      destroy();
      throw std::runtime_error("ikt: USM allocation failed");
    }
    // Pre-touch the event buffer so first-write page faults are not charged
    // to the instrumented kernel (they otherwise appear as a one-tile stall).
    queue_.memset(d_events_, 0, events * sizeof(Event)).wait_and_throw();
    reset();
  }

  void calibrate() { ns_per_tick_ = detail::calibrate_ns_per_tick(queue_); }

  void reset() {
    if (d_counters_) {
      const size_t slots = size_t(wg_count_) * size_t(sgs_per_wg_);
      queue_.memset(d_counters_, 0, slots * sizeof(uint32_t)).wait_and_throw();
    }
  }

  GlobalBuffer global_buffer() const {
    return GlobalBuffer{d_events_, d_counters_, cap_, sgs_per_wg_,
                        wg_base_, wg_count_};
  }

  void set_region_names(std::vector<std::string> names) {
    region_names_ = std::move(names);
  }

  bool write_raw(std::string const& path) const {
    if (!d_events_ || !d_counters_) {
      return false;
    }
    queue_.wait_and_throw();

    const size_t slots = size_t(wg_count_) * size_t(sgs_per_wg_);
    const size_t all_events = slots * size_t(cap_);
    std::vector<uint32_t> counters(slots);
    std::vector<Event> events(all_events);
    queue_.memcpy(counters.data(), d_counters_, slots * sizeof(uint32_t))
        .wait_and_throw();
    queue_.memcpy(events.data(), d_events_, all_events * sizeof(Event))
        .wait_and_throw();

    std::vector<uint32_t> valid_counts(slots);
    std::vector<Event> valid_events;
    valid_events.reserve(all_events);
    for (size_t slot = 0; slot < slots; ++slot) {
      const uint32_t n = std::min(counters[slot], cap_);
      const uint32_t start = counters[slot] > cap_ ? counters[slot] - cap_ : 0;
      valid_counts[slot] = n;
      for (uint32_t i = 0; i < n; ++i) {
        const uint32_t idx = (start + i) & (cap_ - 1u);
        valid_events.push_back(events[slot * size_t(cap_) + idx]);
      }
    }

    double ns_per_tick = ns_per_tick_;
    if (!(ns_per_tick > 0.0)) {
      ns_per_tick = detail::calibrate_ns_per_tick(queue_);
    }
    if (!(ns_per_tick > 0.0)) {
      std::fprintf(stderr, "ikt: clock calibration failed\n");
      return false;
    }

    std::ofstream out(path, std::ios::binary);
    if (!out) {
      std::fprintf(stderr, "ikt: cannot open %s\n", path.c_str());
      return false;
    }

    const char magic[8] = {'I', 'K', 'T', '0', '3', '\0', '\0', '\0'};
    out.write(magic, sizeof(magic));
    const uint32_t version = 3;
    const uint32_t num_regions = uint32_t(region_names_.size());
    const uint32_t event_size = uint32_t(sizeof(Event));
    const uint32_t reserved = 0;
    out.write(reinterpret_cast<const char*>(&version), sizeof(version));
    out.write(reinterpret_cast<const char*>(&cap_), sizeof(cap_));
    out.write(reinterpret_cast<const char*>(&grid_), sizeof(grid_));
    out.write(reinterpret_cast<const char*>(&wg_base_), sizeof(wg_base_));
    out.write(reinterpret_cast<const char*>(&wg_count_), sizeof(wg_count_));
    out.write(reinterpret_cast<const char*>(&sgs_per_wg_), sizeof(sgs_per_wg_));
    out.write(reinterpret_cast<const char*>(&num_regions), sizeof(num_regions));
    out.write(reinterpret_cast<const char*>(&event_size), sizeof(event_size));
    out.write(reinterpret_cast<const char*>(&reserved), sizeof(reserved));
    out.write(reinterpret_cast<const char*>(&ns_per_tick), sizeof(ns_per_tick));

    for (auto const& name : region_names_) {
      const uint32_t len = uint32_t(name.size());
      out.write(reinterpret_cast<const char*>(&len), sizeof(len));
      out.write(name.data(), len);
    }
    out.write(reinterpret_cast<const char*>(valid_counts.data()),
              valid_counts.size() * sizeof(uint32_t));
    out.write(reinterpret_cast<const char*>(valid_events.data()),
              valid_events.size() * sizeof(Event));
    return out.good();
  }

  uint32_t cap() const { return cap_; }
  uint32_t grid_size() const { return grid_; }
  uint32_t wg_base() const { return wg_base_; }
  uint32_t wg_count() const { return wg_count_; }
  uint32_t sgs_per_wg() const { return sgs_per_wg_; }
  std::vector<std::string> const& region_names() const { return region_names_; }

 private:
  void destroy() {
    if (d_events_) {
      sycl::free(d_events_, queue_);
      d_events_ = nullptr;
    }
    if (d_counters_) {
      sycl::free(d_counters_, queue_);
      d_counters_ = nullptr;
    }
  }

  sycl::queue& queue_;
  Event* d_events_ = nullptr;
  uint32_t* d_counters_ = nullptr;
  uint32_t cap_ = 0;
  uint32_t grid_ = 0;
  uint32_t wg_base_ = 0;
  uint32_t wg_count_ = 0;
  uint32_t sgs_per_wg_ = 0;
  double ns_per_tick_ = 0.0;
  std::vector<std::string> region_names_;
};

}  // namespace ikt
