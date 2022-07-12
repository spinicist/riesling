#pragma once

#include "info.h"
#include "log.h"

namespace rl {

/** Crop functions
 */
template <typename T>
decltype(auto) CropLast2(T &&x, Sz3 const &sz)
{
  Sz3 const fullSz = x.dimensions();
  Sz3 const st = Sz3{0, (fullSz[1] - (sz[1] - 1)) / 2, (fullSz[2] - (sz[2] - 1)) / 2};
  return x.slice(st, sz);
}

template <typename T>
decltype(auto) Crop3(T &&x, Sz3 const &sz)
{
  Sz3 const fullSz = x.dimensions();
  Sz3 const st = Sz3{(fullSz[0] - (sz[0] - 1)) / 2, (fullSz[1] - (sz[1] - 1)) / 2, (fullSz[2] - (sz[2] - 1)) / 2};
  return x.slice(st, Sz3{sz[0], sz[1], sz[2]});
}

template <typename T>
decltype(auto) Crop4(T &&x, Sz3 const &sz)
{
  Sz4 const fullSz = x.dimensions();
  Eigen::IndexList<Eigen::type2index<0>, int, int, int> st;
  st.set(1, (fullSz[1] - (sz[0] - 1)) / 2);
  st.set(2, (fullSz[2] - (sz[1] - 1)) / 2);
  st.set(3, (fullSz[3] - (sz[2] - 1)) / 2);
  return x.slice(st, Sz4{x.dimension(0), sz[0], sz[1], sz[2]});
}

template <typename T>
decltype(auto) CropLast3(T &&x, Sz3 const &crop)
{
  auto xsz = x.dimensions();
  decltype(xsz) st, sz;

  Index const N = x.NumDimensions;
  std::copy_n(xsz.begin(), N - 3, sz.begin());
  std::copy_n(crop.begin(), 3, sz.end() - 3);

  std::fill_n(st.begin(), N - 3, 0);
  std::transform(
    xsz.end() - 3, xsz.end(), crop.begin(), st.end() - 3, [](Index big, Index small) { return (big - small + 1) / 2; });
  return x.slice(st, sz);
}

/** Cropper object - useful for when the same cropping operation will be carried out multiple times
 *
 */
struct Cropper
{
  Cropper(Sz3 const &fullSz, Eigen::Array3l const &cropSz);
  Cropper(Sz3 const &fullSz, Sz3 const &cropSz);
  Cropper(Info const &info, Sz3 const &fullSz, float const extent);
  Sz3 size() const;
  Sz3 start() const;
  Sz4 dims(Index const nChan) const;
  Cx3 newImage() const;
  Cx4 newMultichannel(Index const nChan) const;
  Cx4 newSeries(Index const nVols) const;
  R3 newRealImage() const;
  R4 newRealSeries(Index const nVols) const;

  template <typename T>
  decltype(auto) crop3(T &&x) const
  {
    return x.slice(Sz3{st_[0], st_[1], st_[2]}, Sz3{sz_[0], sz_[1], sz_[2]});
  }

  template <typename T>
  decltype(auto) crop4(T &&x) const
  {
    Eigen::IndexList<Eigen::type2index<0>, int, int, int> st;
    st.set(1, st_[0]);
    st.set(2, st_[1]);
    st.set(3, st_[2]);

    return x.slice(st, Sz4{x.dimension(0), sz_[0], sz_[1], sz_[2]});
  }

  template <typename T>
  decltype(auto) crop5(T &&x) const
  {
    return x.slice(Sz5{0, 0, st_[0], st_[1], st_[2]}, Sz5{x.dimension(0), x.dimension(1), sz_[0], sz_[1], sz_[2]});
  }

private:
  Sz3 sz_, st_;
  void calcStart(Sz3 const &fullSz);
};

} // namespace rl
