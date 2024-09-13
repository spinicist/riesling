#include "io/writer.hpp"

#include "io/hd5-core.hpp"
#include "log.hpp"
#include <hdf5.h>
#include <hdf5_hl.h>

namespace rl {
namespace HD5 {

Writer::Writer(std::string const &fname)
{
  Init();
  handle_ = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (handle_ < 0) {
    Log::Fail("HD5", "Could not open file {} for writing", fname);
  } else {
    Log::Print("HD5", "Writing {} id {}", fname, handle_);
  }
}

Writer::~Writer()
{
  H5Fclose(handle_);
  Log::Debug("HD5", "Closed id {}", handle_);
}

void Writer::writeString(std::string const &label, std::string const &string)
{
  herr_t      status;
  hsize_t     dim[1] = {1};
  auto const  space = H5Screate_simple(1, dim, NULL);
  hid_t const tid = H5Tcopy(H5T_C_S1);
  H5Tset_size(tid, H5T_VARIABLE);
  H5Tset_cset(tid, H5T_CSET_UTF8);
  hid_t const dset = H5Dcreate(handle_, label.c_str(), tid, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  auto        ptr = string.c_str();
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ptr);
  status = H5Dclose(dset);
  status = H5Sclose(space);
  if (status) { Log::Fail("HD5", "Could not write string {} into handle {}, code: {}", label, handle_, status); }
}

void Writer::writeInfo(Info const &info)
{
  hid_t       info_id = InfoType();
  hsize_t     dims[1] = {1};
  auto const  space = H5Screate_simple(1, dims, NULL);
  hid_t const dset = H5Dcreate(handle_, Keys::Info.c_str(), info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) { Log::Fail("HD5", "Could not create info struct in file {}, code: {}", handle_, dset); }
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) { Log::Fail("HD5", "Could not write info struct in file {}, code: {}", handle_, status); }
  Log::Debug("HD5", "Wrote info struct");
}

void Writer::writeMeta(std::map<std::string, float> const &meta)
{
  Log::Debug("HD5", "Writing meta data");
  auto m_group = H5Gcreate(handle_, Keys::Meta.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t    dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  herr_t     status;
  for (auto const &kvp : meta) {
    Log::Debug("HD5", "Writing {}:{}", kvp.first, kvp.second);
    hid_t const dset = H5Dcreate(m_group, kvp.first.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(kvp.second));
    status = H5Dclose(dset);
  }
  status = H5Sclose(space);
  status = H5Gclose(m_group);
  if (status != 0) { Log::Fail("HD5", "Exception occured storing meta-data in file {}", handle_); }
}

bool Writer::exists(std::string const &name) const { return HD5::Exists(handle_, name); }

template <typename Scalar, int N>
void Writer::writeTensor(std::string const &name, Sz<N> const &shape, Scalar const *data, DimensionNames<N> const &labels)
{
  for (Index ii = 0; ii < N; ii++) {
    if (shape[ii] == 0) { Log::Fail("HD5", "Tensor {} had a zero dimension. Dims: {}", name, shape); }
  }

  hsize_t ds_dims[N], chunk_dims[N];
  // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
  std::copy_n(shape.rbegin(), N, ds_dims);
  std::copy_n(ds_dims, N, chunk_dims);
  // Try to stop chunk dimension going over 4 gig
  Index sizeInBytes = Product(shape) * sizeof(Scalar);
  Index dimToShrink = 0;
  while (sizeInBytes > (1L << 32L)) {
    if (chunk_dims[dimToShrink] > 1) {
      chunk_dims[dimToShrink] /= 2;
      sizeInBytes /= 2;
    }
    dimToShrink = (dimToShrink + 1) % N;
  }

  auto const space = H5Screate_simple(N, ds_dims, NULL);
  auto const plist = H5Pcreate(H5P_DATASET_CREATE);
  CheckedCall(H5Pset_deflate(plist, 2), "setting deflate");
  CheckedCall(H5Pset_chunk(plist, N, chunk_dims), "setting chunk");

  hid_t const tid = type<Scalar>();
  hid_t const dset = H5Dcreate(handle_, name.c_str(), tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail("HD5", "Could not create dataset {} dimensions {} error {}", name, shape, GetError());
  }
  auto        l = labels.rbegin();
  for (Index ii = 0; ii < N; ii++) {
    CheckedCall(H5DSset_label(dset, ii, l->c_str()), fmt::format("dataset {} dimension {} label {}", name, ii, *l));
    l++;
  }
  if (dset < 0) { Log::Fail("HD5", "Could not create tensor {}. Dims {}. Error {}", name, fmt::join(shape, ","), HD5::GetError()); }
  CheckedCall(H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "Writing data");
  CheckedCall(H5Pclose(plist), "closing plist");
  CheckedCall(H5Sclose(space), "closing space");
  CheckedCall(H5Dclose(dset), "closing dataset");

  Log::Debug("HD5", "Wrote tensor {}", name);
}

template void Writer::writeTensor<Index, 1>(std::string const &, Sz<1> const &, Index const *, DimensionNames<1> const &);
template void Writer::writeTensor<float, 1>(std::string const &, Sz<1> const &, float const *, DimensionNames<1> const &);
template void Writer::writeTensor<float, 2>(std::string const &, Sz<2> const &, float const *, DimensionNames<2> const &);
template void Writer::writeTensor<float, 3>(std::string const &, Sz<3> const &, float const *, DimensionNames<3> const &);
template void Writer::writeTensor<float, 4>(std::string const &, Sz<4> const &, float const *, DimensionNames<4> const &);
template void Writer::writeTensor<float, 5>(std::string const &, Sz<5> const &, float const *, DimensionNames<5> const &);
template void Writer::writeTensor<Cx, 2>(std::string const &, Sz<2> const &, Cx const *, DimensionNames<2> const &);
template void Writer::writeTensor<Cx, 3>(std::string const &, Sz<3> const &, Cx const *, DimensionNames<3> const &);
template void Writer::writeTensor<Cx, 4>(std::string const &, Sz<4> const &, Cx const *, DimensionNames<4> const &);
template void Writer::writeTensor<Cx, 5>(std::string const &, Sz<5> const &, Cx const *, DimensionNames<5> const &);
template void Writer::writeTensor<Cx, 6>(std::string const &, Sz<6> const &, Cx const *, DimensionNames<6> const &);

template <typename Derived>
void Writer::writeMatrix(Eigen::DenseBase<Derived> const &mat, std::string const &name)
{
  herr_t        status;
  hsize_t       ds_dims[2], chunk_dims[2];
  hsize_t const rank = mat.cols() > 1 ? 2 : 1;
  if (rank == 2) {
    // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
    ds_dims[0] = mat.cols();
    ds_dims[1] = mat.rows();
  } else {
    ds_dims[0] = mat.rows();
  }

  std::copy_n(ds_dims, rank, chunk_dims);
  auto const space = H5Screate_simple(rank, ds_dims, NULL);
  auto const plist = H5Pcreate(H5P_DATASET_CREATE);
  status = H5Pset_deflate(plist, rank);
  status = H5Pset_chunk(plist, rank, chunk_dims);

  hid_t const tid = type<typename Derived::Scalar>();
  hid_t const dset = H5Dcreate(handle_, name.c_str(), tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, mat.derived().data());
  status = H5Pclose(plist);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status) {
    Log::Fail("HD5", "Could not write matrix {} into handle {}, code: {}", name, handle_, status);
  } else {
    Log::Debug("HD5", "Wrote matrix: {}", name);
  }
}

template void Writer::writeMatrix<Eigen::MatrixXf>(Eigen::DenseBase<Eigen::MatrixXf> const &, std::string const &);
template void Writer::writeMatrix<Eigen::MatrixXcf>(Eigen::DenseBase<Eigen::MatrixXcf> const &, std::string const &);

template void Writer::writeMatrix<Eigen::ArrayXf>(Eigen::DenseBase<Eigen::ArrayXf> const &, std::string const &);
template void Writer::writeMatrix<Eigen::ArrayXXf>(Eigen::DenseBase<Eigen::ArrayXXf> const &, std::string const &);

template <typename T>
auto writeAttribute(std::string const &dataset, std::string const &attribute, T const &val);

template <int N>
void Writer::writeAttribute(std::string const &dset, std::string const &attr, Sz<N> const &val)
{
  hsize_t const szN[1] = {N}, sz1[1] = {1};
  hid_t const   long3_id = H5Tarray_create(H5T_NATIVE_LONG, 1, szN);
  auto const    space = H5Screate_simple(1, sz1, NULL);
  auto const    attrH =
    H5Acreate_by_name(handle_, dset.c_str(), attr.c_str(), long3_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  CheckedCall(H5Awrite(attrH, long3_id, val.data()), fmt::format("writing attribute {} to {}", attr, dset));
}

template void Writer::writeAttribute<1>(std::string const &dset, std::string const &attr, Sz<1> const &val);
template void Writer::writeAttribute<2>(std::string const &dset, std::string const &attr, Sz<2> const &val);
template void Writer::writeAttribute<3>(std::string const &dset, std::string const &attr, Sz<3> const &val);



} // namespace HD5
} // namespace rl
