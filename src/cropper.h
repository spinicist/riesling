#pragma once

#include "info.hpp"
#include "log.hpp"

namespace rl {

template <typename T, int ND>
decltype(auto) Crop(Eigen::Tensor<T, ND> const &x, Sz<ND> const &sz)
{
  Sz<ND> const fullSz = x.dimensions();
  Sz<ND>       st;
  for (Index ii = 0; ii < ND; ii++) {
    if (sz[ii] > fullSz[ii]) {
      Log::Fail("Requested crop dimensions {} exceeded tensor dimensions {}", sz, fullSz);
    }
    st[ii] = (fullSz[ii] - (sz[ii] - 1)) / 2;
  }
  return x.slice(st, sz);
}

template <typename T, int ND>
decltype(auto) Crop(Eigen::Tensor<T, ND> &x, Sz<ND> const &sz)
{
  Sz<ND> const fullSz = x.dimensions();
  Sz<ND>       st;
  for (Index ii = 0; ii < ND; ii++) {
    if (sz[ii] > fullSz[ii]) {
      Log::Fail("Requested crop dimensions {} exceeded tensor dimensions {}", sz, fullSz);
    }
    st[ii] = (fullSz[ii] - (sz[ii] - 1)) / 2;
  }
  return x.slice(st, sz);
}


/** Cropper object - useful for when the same cropping operation will be carried out multiple times
 *
 */
struct Cropper
{
  Cropper(Sz3 const &fullSz, Eigen::Array3l const &cropSz);
  Cropper(Sz3 const &fullSz, Sz3 const &cropSz);
  Cropper(Sz3 const matrix, Sz3 const fullSz, Eigen::Array3f const voxelSz, Eigen::Array3f const extent);
  Sz3 size() const;
  Sz3 start() const;
  Sz4 dims(Index const nChan) const;
  Cx3 newImage() const;
  Cx4 newMultichannel(Index const nChan) const;
  Cx4 newSeries(Index const nVols) const;
  Re3 newRealImage() const;
  Re4 newRealSeries(Index const nVols) const;

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
  Sz3  sz_, st_;
  void calcStart(Sz3 const &fullSz);
};

} // namespace rl
