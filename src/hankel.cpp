#include "hankel.h"

#include "tensorOps.h"

Cx5 ToKernels(Cx4 const &grid, long const kRad, long const calRad, long const gapRad, Log &log)
{
  long const nchan = grid.dimension(0);
  long const gridHalf = grid.dimension(1) / 2;
  long const calW = (calRad * 2) - 1;
  long const kW = (kRad * 2) - 1;
  long const gapPlusKW = ((gapRad + kRad) * 2) - 1;
  long const nSkip = gapRad ? gapPlusKW * gapPlusKW * gapPlusKW : 0;
  long const nk = calW * calW * calW - nSkip;
  if (nk < 1) {
    Log::Fail(FMT_STRING("No kernels to Hankelfy"));
  }
  Cx5 kernels(nchan, kW, kW, kW, nk);

  long k = 0;
  long s = 0;
  long const gapSt = (calRad - 1) - (gapRad - 1) - kRad;
  long const gapEnd = (calRad - 1) + gapRad + kRad;

  long const st = gridHalf - (calRad - 1) - (kRad - 1);
  if (st < 0) {
    Log::Fail(
        FMT_STRING("Grid size {} not large enough for calibration radius {} + kernel radius {}"),
        grid.dimension(1),
        calRad,
        kRad);
  }

  log.info(
      FMT_STRING("Hankel calibration rad {} kernel rad {} gap {}, {} kernels"),
      calRad,
      kRad,
      gapRad,
      nk);
  for (long iz = 0; iz < calW; iz++) {
    long const st_z = st + iz;
    for (long iy = 0; iy < calW; iy++) {
      long const st_y = st + iy;
      for (long ix = 0; ix < calW; ix++) {
        if (gapRad && (ix >= gapSt && ix < gapEnd) && (iy >= gapSt && iy < gapEnd) &&
            (iz >= gapSt && iz < gapEnd)) {
          s++;
          // fmt::print("SKIP {} {} {} GAP {} {}\n", iz, iy, ix, gapSt, gapEnd);
          continue;
        }

        long const st_x = st + ix;
        Sz4 sst{0, st_x, st_y, st_z};
        Sz4 ssz{nchan, kW, kW, kW};
        // fmt::print(
        //     "USE  {} {} {} GAP {} {} ST {} SZ {} k {} dim {}\n",
        //     iz,
        //     iy,
        //     ix,
        //     gapSt,
        //     gapEnd,
        //     fmt::join(sst, ","),
        //     fmt::join(ssz, ","),
        //     k,
        //     kernels.dimension(4));
        kernels.chip(k, 4) = grid.slice(sst, ssz);
        k++;
      }
    }
  }
  assert(s == nSkip);
  assert(k == nk);
  return kernels;
}

void FromKernels(long const calSz, long const kSz, Cx2 const &kernels, Cx4 &grid, Log &log)
{
  long const nchan = grid.dimension(0);
  long const gridHalf = grid.dimension(1) / 2;
  long const calHalf = calSz / 2;
  long const kHalf = kSz / 2;
  if (grid.dimension(1) < (calSz + kSz - 1)) {
    Log::Fail(
        FMT_STRING("Grid size {} not large enough for block size {} + kernel size {}"),
        grid.dimension(1),
        calSz,
        kSz);
  }
  assert(kernels.dimension(0) == nchan * kSz * kSz * kSz);
  assert(kernels.dimension(1) == calSz * calSz * calSz);

  long const st = gridHalf - calHalf - kHalf;
  long const sz = calSz + kSz - 1;
  R3 count(sz, sz, sz);
  count.setZero();
  Cx4 data(nchan, sz, sz, sz);
  data.setZero();
  long col = 0;
  for (long iz = 0; iz < calSz; iz++) {
    for (long iy = 0; iy < calSz; iy++) {
      for (long ix = 0; ix < calSz; ix++) {
        data.slice(Sz4{0, ix, iy, iz}, Sz4{nchan, kSz, kSz, kSz}) +=
            kernels.chip(col, 1).reshape(Sz4{nchan, kSz, kSz, kSz});
        count.slice(Sz3{ix, iy, iz}, Sz3{kSz, kSz, kSz}) +=
            count.slice(Sz3{ix, iy, iz}, Sz3{kSz, kSz, kSz}).constant(1.f);
        col++;
      }
    }
  }
  assert(col == calSz * calSz * calSz);
  grid.slice(Sz4{0, st, st, st}, Sz4{nchan, sz, sz, sz}) =
      data.abs().select(data / Tile(count, nchan).cast<Cx>(), data);
}