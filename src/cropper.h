#pragma once

#include "info.h"
#include "log.h"

/** Crop functions
 */
template <typename T>
decltype(auto) Crop3(T &&x, Sz3 const &sz)
{
  Dims3 const fullSz = x.dimensions();
  Dims3 const st = Dims3{
      (fullSz[0] - (sz[0] - 1)) / 2, (fullSz[1] - (sz[1] - 1)) / 2, (fullSz[2] - (sz[2] - 1)) / 2};
  return x.slice(Dims3{st[0], st[1], st[2]}, Dims3{sz[0], sz[1], sz[2]});
}

template <typename T>
decltype(auto) Crop4(T &&x, Dims3 const &sz)
{
  Dims4 const fullSz = x.dimensions();
  Dims3 const st = Dims3{
      (fullSz[1] - (sz[0] - 1)) / 2, (fullSz[2] - (sz[1] - 1)) / 2, (fullSz[3] - (sz[2] - 1)) / 2};
  return x.slice(Dims4{0, st[0], st[1], st[2]}, Dims4{x.dimension(0), sz[0], sz[1], sz[2]});
}

/** Cropper object - useful for when the same cropping operation will be carried out multiple times
 *
 */
struct Cropper
{
  Cropper(Dims3 const &fullSz, Dims3 const &cropSz, Log &log);
  Cropper(Dims3 const &fullSz, Array3l const &cropSz, Log &log);
  Cropper(Dims3 const &fullSz, Sz3 const &cropSz, Log &log);
  Cropper(Info const &info, Dims3 const &fullSz, float const extent, Log &log);
  Dims3 size() const;
  Dims3 start() const;
  Cx3 newImage() const;
  Cx4 newMultichannel(long const nChan) const;
  Cx4 newSeries(long const nVols) const;
  R3 newRealImage() const;
  R4 newRealSeries(long const nVols) const;

  template <typename T>
  decltype(auto) crop3(T &&x) const
  {
    return x.slice(Dims3{st_[0], st_[1], st_[2]}, Dims3{sz_[0], sz_[1], sz_[2]});
  }

  template <typename T>
  decltype(auto) crop4(T &&x) const
  {
    return x.slice(Dims4{0, st_[0], st_[1], st_[2]}, Dims4{x.dimension(0), sz_[0], sz_[1], sz_[2]});
  }

private:
  Dims3 sz_, st_;
  void calcStart(Dims3 const &fullSz);
};
