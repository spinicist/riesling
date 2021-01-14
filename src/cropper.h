#pragma once

#include "info.h"
#include "log.h"

template <typename T>
decltype(auto) Crop4(T &&x, Dims3 const &cropSz)
{
  Dims4 const &fullSz = x.dimensions();
  Dims4 const start{0,
                    (fullSz[1] - cropSz[0]) / 2 + 1,
                    (fullSz[2] - cropSz[1]) / 2 + 1,
                    (fullSz[3] - cropSz[2]) / 2 + 1};
  Dims4 const size{fullSz[0], cropSz[0], cropSz[1], cropSz[2]};
  return x.slice(start, size);
}

struct Cropper
{
  Cropper(Info const &info, Dims3 const &fullSz, float const extent, bool const crop, Log &log);
  Cropper(Info const &info, Dims3 const &fullSz, Dims3 const &cropSz, Log &log);
  Dims3 size() const;
  Dims3 start() const;
  Cx3 newImage() const;
  Cx4 newMultichannel(long const nChan) const;
  Cx4 newSeries(long const nVols) const;
  R4 newRealSeries(long const nVols) const;

  template <typename T>
  decltype(auto) crop3(T &&x)
  {
    return x.slice(Dims3{st_[0], st_[1], st_[2]}, Dims3{sz_[0], sz_[1], sz_[2]});
  }

  template <typename T>
  decltype(auto) crop4(T &&x)
  {
    return x.slice(Dims4{0, st_[0], st_[1], st_[2]}, Dims4{x.dimension(0), sz_[0], sz_[1], sz_[2]});
  }

private:
  Dims3 sz_, st_;
};
