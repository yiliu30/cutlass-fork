#pragma once

// Compatibility shim: the FMHA worktree predates the ikt rename. Map the old
// ikp_xe names and macros onto the aligned Intra Kernel Tracer backend.
#ifndef IKT_TRACE_ENABLED
#  ifdef IKP_XE_TRACE_ENABLED
#    define IKT_TRACE_ENABLED IKP_XE_TRACE_ENABLED
#  else
#    define IKT_TRACE_ENABLED 1
#  endif
#endif
#include "ikt/trace.hpp"

namespace ikp_xe {
using GlobalBuffer = ikt::GlobalBuffer;
template <uint32_t Cap>
using SubgroupContext = ikt::SubgroupContext<Cap>;
using HostSession = ikt::HostSession;
}  // namespace ikp_xe

#define IKP_XE_CTX_TYPE(cap) IKT_CTX_TYPE(cap)
#define IKP_XE_CTX_INIT(ctx, gbuf, item, sg) IKT_CTX_INIT(ctx, gbuf, item, sg)
#define IKP_XE_BEGIN(ctx, gbuf, sg, id) IKT_BEGIN(ctx, gbuf, sg, id)
#define IKP_XE_END(ctx, gbuf, sg, id) IKT_END(ctx, gbuf, sg, id)
#define IKP_XE_MARK(ctx, gbuf, sg, id) IKT_MARK(ctx, gbuf, sg, id)
#define IKP_XE_BEGIN_IF(ctx, gbuf, sg, id, cond) IKT_BEGIN_IF(ctx, gbuf, sg, id, cond)
#define IKP_XE_END_IF(ctx, gbuf, sg, id, cond) IKT_END_IF(ctx, gbuf, sg, id, cond)
#define IKP_XE_FLUSH(ctx, gbuf) IKT_FLUSH(ctx, gbuf)
