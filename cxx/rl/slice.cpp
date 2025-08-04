#include "slice.hpp"

#include "log/log.hpp"

namespace rl {

auto SliceNC(Sz3 const   channel,
             Sz3 const   sample,
             Sz3 const   trace,
             Sz3 const   slab,
             Sz3 const   time,
             Index const tps,
             Sz2 const   segment,
             Cx5 const  &ks,
             Re3 const  &trajPoints) -> SliceNCT
{
  auto const  shape = ks.dimensions();
  Index const cSt = Wrap(channel[0], shape[0]);
  Index const rSt = Wrap(sample[0], shape[1]);
  Index const tSt = Wrap(trace[0], shape[2]);
  Index const sSt = Wrap(slab[0], shape[3]);
  Index const uSt = Wrap(time[0], shape[4]);

  Index const cSz = channel[1] > 0 ? channel[1] : shape[0] - cSt;
  Index const rSz = sample[1] > 0 ? sample[1] : shape[1] - rSt;
  Index const tSz = trace[1] > 0 ? trace[1] : (tps ? tps - tSt : shape[2] - tSt);
  Index const sSz = slab[1] > 0 ? slab[1] : shape[3] - sSt;
  Index const uSz = time[1] > 0 ? time[1] : shape[4] - uSt;

  if (cSt + cSz > shape[0]) { throw Log::Failure("slice", "Last channel {} exceeded maximum {}", cSt + cSz, shape[0]); }
  if (rSt + rSz > shape[1]) { throw Log::Failure("slice", "Last sample {} exceeded maximum {}", rSt + rSz, shape[1]); }
  if (tps) {
    if (tSt + tSz > tps) { throw Log::Failure("slice", "Last trace {} exceeded segment size {}", tSt + tSz, tps); }
  } else {
    if (tSt + tSz > shape[2]) { throw Log::Failure("slice", "Last trace {} exceeded maximum {}", tSt + tSz, shape[2]); }
  }
  if (sSt + sSz > shape[3]) { throw Log::Failure("slice", "Last slab {} exceeded maximum {}", sSt + sSz, shape[3]); }
  if (uSt + uSz > shape[4]) { throw Log::Failure("slice", "Last volume {} exceeded maximum {}", tSt + tSz, shape[4]); }

  if (cSz < 1) { throw Log::Failure("slice", "Channel size was less than 1"); }
  if (rSz < 1) { throw Log::Failure("slice", "Sample size was less than 1"); }
  if (tSz < 1) { throw Log::Failure("slice", "Trace size was less than 1"); }
  if (sSz < 1) { throw Log::Failure("slice", "Slab size was less than 1"); }
  if (uSz < 1) { throw Log::Failure("slice", "Volume size was less than 1"); }

  Log::Print("slice", "Selected slice {}:{}, {}:{}, {}:{}, {}:{}, {}:{}", cSt, cSt + cSz - 1, rSt, rSt + rSz - 1, tSt,
             tSt + tSz - 1, sSt, sSt + sSz - 1, uSt, uSt + uSz - 1);

  Cx5 sks;
  Re3 stp;
  if (tps) {
    if (tSt + tSz > tps) { throw Log::Failure("slice", "Selected traces {}-{} extend past segment {}", tSt, tSz, tps); }
    Index const nSeg = shape[2] / tps;
    if (nSeg * tps != shape[2]) {
      Log::Warn("slice", "Traces per seg {} does not cleanly divide traces {}, dropping last segment", tps, shape[2]);
    }
    Index const segSt = Wrap(segment[0], nSeg);
    Index const segSz = segment[1] > 0 ? std::clamp(segment[1], 1L, nSeg) : nSeg - segSt;
    Log::Print("slice", "Selected segments {}:{}", segSt, segSt + segSz - 1);
    auto segs = ks.slice(Sz5{}, Sz5{shape[0], shape[1], tps * nSeg, shape[3], shape[4]})
                  .reshape(Sz6{shape[0], shape[1], tps, nSeg, shape[3], shape[4]});
    auto sliced = segs.slice(Sz6{cSt, rSt, tSt, segSt, sSt, uSt}, Sz6{cSz, rSz, tSz, segSz, sSz, uSz});
    sks = sliced.reshape(Sz5{cSz, rSz, tSz * segSz, sSz, uSz});
    auto tsegs = trajPoints.reshape(Sz4{3, trajPoints.dimension(1), tps, nSeg});
    auto tsliced = tsegs.slice(Sz4{0, rSt, tSt, segSt}, Sz4{3, rSz, tSz, segSz});
    stp = tsliced.reshape(Sz3{3, rSz, tSz * segSz});
  } else {
    sks = ks.slice(Sz5{cSt, rSt, tSt, sSt, uSt}, Sz5{cSz, rSz, tSz, sSz, uSz});
    stp = trajPoints.slice(Sz3{0, rSt, tSt}, Sz3{3, rSz, tSz});
  }

  if ((channel[2] > 1) || (sample[2] > 1) || (trace[2] > 1) || (slab[2] > 1) || (time[2] > 1)) {
    sks = Cx5(sks.stride(Sz5{channel[2], sample[2], trace[2], slab[2], time[2]}));
    stp = Re3(stp.stride(Sz3{1, sample[2], trace[2]}));
  }
  return {sks, stp};
}

} // namespace rl
