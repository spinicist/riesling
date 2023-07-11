#include "io/writer.hpp"

#include "io/hd5-core.hpp"
#include "log.hpp"
#include <hdf5.h>

namespace rl {
namespace HD5 {

template <typename Scalar, int ND>
void store_tensor(Handle const &parent, std::string const &name, Sz<ND> const &dims, Scalar const *data)
{
  // Check for sane dimensions
  for (Index ii = 0; ii < ND; ii++) {
    if (dims[ii] == 0) { Log::Fail("Tensor {} had a zero dimension. Dims: {}", name, dims); }
  }

  herr_t  status;
  hsize_t ds_dims[ND], chunk_dims[ND];
  // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
  std::copy_n(dims.rbegin(), ND, ds_dims);
  std::copy_n(ds_dims, ND, chunk_dims);
  // Try to stop chunk dimension going over 4 gig
  Index sizeInBytes = Product(dims) * sizeof(Scalar);
  Index dimToShrink = 0;
  while (sizeInBytes > (1L << 32L)) {
    if (chunk_dims[dimToShrink] > 1) {
      chunk_dims[dimToShrink] /= 2;
      sizeInBytes /= 2;
    }
    dimToShrink = (dimToShrink + 1) % ND;
  }

  auto const space = H5Screate_simple(ND, ds_dims, NULL);
  auto const plist = H5Pcreate(H5P_DATASET_CREATE);
  status = H5Pset_deflate(plist, 2);
  status = H5Pset_chunk(plist, ND, chunk_dims);

  hid_t const tid = type<Scalar>();
  hid_t const dset = H5Dcreate(parent, name.c_str(), tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not create tensor {}. Dims {}. Error {}", name, fmt::join(dims, ","), HD5::GetError()); }
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);
  status = H5Pclose(plist);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status) {
    Log::Fail("Writing Tensor {}: Error {}", name, HD5::GetError());
  } else {
    Log::Print<Log::Level::High>("Wrote tensor: {}", name);
  }
}

template <typename Derived>
void store_matrix(Handle const &parent, std::string const &name, Eigen::DenseBase<Derived> const &data)
{
  herr_t        status;
  hsize_t       ds_dims[2], chunk_dims[2];
  hsize_t const rank = data.cols() > 1 ? 2 : 1;
  if (rank == 2) {
    // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
    ds_dims[0] = data.cols();
    ds_dims[1] = data.rows();
  } else {
    ds_dims[0] = data.rows();
  }

  std::copy_n(ds_dims, rank, chunk_dims);
  auto const space = H5Screate_simple(rank, ds_dims, NULL);
  auto const plist = H5Pcreate(H5P_DATASET_CREATE);
  status = H5Pset_deflate(plist, rank);
  status = H5Pset_chunk(plist, rank, chunk_dims);

  hid_t const tid = type<typename Derived::Scalar>();
  hid_t const dset = H5Dcreate(parent, name.c_str(), tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.derived().data());
  status = H5Pclose(plist);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status) {
    Log::Fail("Could not write matrix {} into handle {}, code: {}", name, parent, status);
  } else {
    Log::Print<Log::Level::High>("Wrote matrix: {}", name);
  }
}

Writer::Writer(std::string const &fname)
{
  Init();
  handle_ = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (handle_ < 0) {
    Log::Fail("Could not open file {} for writing", fname);
  } else {
    Log::Print("Opened file to write: {}", fname);
    Log::Print<Log::Level::High>("Handle: {}", handle_);
  }
}

Writer::~Writer()
{
  H5Fclose(handle_);
  Log::Print<Log::Level::High>("Closed handle: {}", handle_);
}

void Writer::writeString(std::string const &label, std::string const &string)
{
  herr_t      status;
  hsize_t     dim[1] = {1};
  auto const  space = H5Screate_simple(1, dim, NULL);
  hid_t const tid = H5Tcopy(H5T_C_S1);
  H5Tset_size(tid, H5T_VARIABLE);
  hid_t const dset = H5Dcreate(handle_, label.c_str(), tid, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  auto        ptr = string.c_str();
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, &ptr);
  status = H5Dclose(dset);
  status = H5Sclose(space);
  if (status) { Log::Fail("Could not write string {} into handle {}, code: {}", label, handle_, status); }
}

void Writer::writeInfo(Info const &info)
{
  hid_t       info_id = InfoType();
  hsize_t     dims[1] = {1};
  auto const  space = H5Screate_simple(1, dims, NULL);
  hid_t const dset = H5Dcreate(handle_, Keys::Info.c_str(), info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) { Log::Fail("Could not create info struct in file {}, code: {}", handle_, dset); }
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) { Log::Fail("Could not write info struct in file {}, code: {}", handle_, status); }
  Log::Print<Log::Level::High>("Wrote info struct");
}

void Writer::writeMeta(std::map<std::string, float> const &meta)
{
  Log::Print("Writing meta data");
  auto m_group = H5Gcreate(handle_, Keys::Meta.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t    dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  herr_t     status;
  for (auto const &kvp : meta) {
    Log::Print("Writing {}:{}", kvp.first, kvp.second);
    hid_t const dset = H5Dcreate(m_group, kvp.first.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(kvp.second));
    status = H5Dclose(dset);
  }
  status = H5Sclose(space);
  status = H5Gclose(m_group);
  if (status != 0) { Log::Fail("Exception occured storing meta-data in file {}", handle_); }
}

bool Writer::exists(std::string const &name) const { return HD5::Exists(handle_, name); }

template <typename Scalar, int ND>
void Writer::writeTensor(std::string const &label, Sz<ND> const &shape, Scalar const *data)
{
  HD5::store_tensor(handle_, label, shape, data);
}

template void Writer::writeTensor<Index, 1>(std::string const &, Sz<1> const &, Index const *);
template void Writer::writeTensor<float, 1>(std::string const &, Sz<1> const &, float const *);
template void Writer::writeTensor<float, 2>(std::string const &, Sz<2> const &, float const *);
template void Writer::writeTensor<float, 3>(std::string const &, Sz<3> const &, float const *);
template void Writer::writeTensor<float, 4>(std::string const &, Sz<4> const &, float const *);
template void Writer::writeTensor<float, 5>(std::string const &, Sz<5> const &, float const *);
template void Writer::writeTensor<Cx, 3>(std::string const &, Sz<3> const &, Cx const *);
template void Writer::writeTensor<Cx, 4>(std::string const &, Sz<4> const &, Cx const *);
template void Writer::writeTensor<Cx, 5>(std::string const &, Sz<5> const &, Cx const *);
template void Writer::writeTensor<Cx, 6>(std::string const &, Sz<6> const &, Cx const *);

template <typename Derived>
void Writer::writeMatrix(Eigen::DenseBase<Derived> const &m, std::string const &label)
{
  HD5::store_matrix(handle_, label, m);
}

template void Writer::writeMatrix<Eigen::MatrixXf>(Eigen::DenseBase<Eigen::MatrixXf> const &, std::string const &);
template void Writer::writeMatrix<Eigen::MatrixXcf>(Eigen::DenseBase<Eigen::MatrixXcf> const &, std::string const &);

template void Writer::writeMatrix<Eigen::ArrayXf>(Eigen::DenseBase<Eigen::ArrayXf> const &, std::string const &);
template void Writer::writeMatrix<Eigen::ArrayXXf>(Eigen::DenseBase<Eigen::ArrayXXf> const &, std::string const &);

} // namespace HD5
} // namespace rl
