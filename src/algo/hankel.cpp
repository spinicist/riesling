#include "hankel.h"

#include "tensorOps.h"

Cx5 ToKernels(Cx4 const &grid, Index const kRad, Index const calRad, Index const gapRad)
{
  Index const nchan = grid.dimension(0);
  Index const gridHalf = grid.dimension(1) / 2;
  Index const calW = (calRad * 2) - 1;
  Index const kW = (kRad * 2) - 1;
  Index const gapPlusKW = ((gapRad + kRad) * 2) - 1;
  Index const nSkip = gapRad ? gapPlusKW * gapPlusKW * gapPlusKW : 0;
  Index const nk = calW * calW * calW - nSkip;
  if (nk < 1) {
    Log::Fail(FMT_STRING("No kernels to Hankelfy"));
  }
  Cx5 kernels(nchan, kW, kW, kW, nk);

  Index k = 0;
  Index s = 0;
  Index const gapSt = (calRad - 1) - (gapRad - 1) - kRad;
  Index const gapEnd = (calRad - 1) + gapRad + kRad;

  Index const st = gridHalf - (calRad - 1) - (kRad - 1);
  if (st < 0) {
    Log::Fail(
      FMT_STRING("Grid size {} not large enough for calibration radius {} + kernel radius {}"),
      grid.dimension(1),
      calRad,
      kRad);
  }

  Log::Print(
    FMT_STRING("Hankel calibration rad {} kernel rad {} gap {}, {} kernels"),
    calRad,
    kRad,
    gapRad,
    nk);
  for (Index iz = 0; iz < calW; iz++) {
    Index const st_z = st + iz;
    for (Index iy = 0; iy < calW; iy++) {
      Index const st_y = st + iy;
      for (Index ix = 0; ix < calW; ix++) {
        if (
          gapRad && (ix >= gapSt && ix < gapEnd) && (iy >= gapSt && iy < gapEnd) &&
          (iz >= gapSt && iz < gapEnd)) {
          s++;
          continue;
        }

        Index const st_x = st + ix;
        Sz4 sst{0, st_x, st_y, st_z};
        Sz4 ssz{nchan, kW, kW, kW};
        kernels.chip<4>(k) = grid.slice(sst, ssz);
        k++;
      }
    }
  }
  assert(s == nSkip);
  assert(k == nk);
  return kernels;
}

void FromKernels(Index const calSz, Index const kSz, Cx2 const &kernels, Cx4 &grid)
{
  Index const nchan = grid.dimension(0);
  Index const gridHalf = grid.dimension(1) / 2;
  Index const calHalf = calSz / 2;
  Index const kHalf = kSz / 2;
  if (grid.dimension(1) < (calSz + kSz - 1)) {
    Log::Fail(
      FMT_STRING("Grid size {} not large enough for block size {} + kernel size {}"),
      grid.dimension(1),
      calSz,
      kSz);
  }
  assert(kernels.dimension(0) == nchan * kSz * kSz * kSz);
  assert(kernels.dimension(1) == calSz * calSz * calSz);

  Index const st = gridHalf - calHalf - kHalf;
  Index const sz = calSz + kSz - 1;
  R3 count(sz, sz, sz);
  count.setZero();
  Cx4 data(nchan, sz, sz, sz);
  data.setZero();
  Index col = 0;
  for (Index iz = 0; iz < calSz; iz++) {
    for (Index iy = 0; iy < calSz; iy++) {
      for (Index ix = 0; ix < calSz; ix++) {
        data.slice(Sz4{0, ix, iy, iz}, Sz4{nchan, kSz, kSz, kSz}) +=
          kernels.chip<1>(col).reshape(Sz4{nchan, kSz, kSz, kSz});
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