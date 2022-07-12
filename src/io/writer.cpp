#include "io/writer.hpp"
#include "io/hd5-core.hpp"

namespace rl {
namespace HD5 {

template <typename Scalar, int ND>
void store_tensor(Handle const &parent, std::string const &name, Eigen::Tensor<Scalar, ND> const &data)
{
  // Check for sane dimensions
  for (Index ii = 0; ii < ND; ii++) {
    if (data.dimension(ii) == 0) {
      Log::Fail(FMT_STRING("Tensor {} had a zero dimension. Dims: {}"), name, data.dimensions());
    }
  }

  herr_t status;
  hsize_t ds_dims[ND], chunk_dims[ND];
  // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
  std::copy_n(data.dimensions().rbegin(), ND, ds_dims);
  std::copy_n(ds_dims, ND, chunk_dims);
  // Try to stop chunk dimension going over 4 gig
  Index sizeInBytes = Product(data.dimensions()) * sizeof(Scalar);
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
  if (dset < 0) {
    Log::Fail(
      FMT_STRING("Could not create tensor {}. Dims {}. Error {}"),
      name,
      fmt::join(data.dimensions(), ","),
      HD5::GetError());
  }
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  status = H5Pclose(plist);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status) {
    Log::Fail(FMT_STRING("Writing Tensor {}: Error {}"), name, HD5::GetError());
  } else {
    Log::Debug(FMT_STRING("Wrote tensor: {}"), name);
  }
}

template <typename Derived>
void store_matrix(Handle const &parent, std::string const &name, Eigen::DenseBase<Derived> const &data)
{
  herr_t status;
  hsize_t ds_dims[2], chunk_dims[2];
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
    Log::Fail(FMT_STRING("Could not write matrix {} into handle {}, code: {}"), name, parent, status);
  } else {
    Log::Debug(FMT_STRING("Wrote matrix: {}"), name);
  }
}

Writer::Writer(std::string const &fname)
{
  Init();
  handle_ = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (handle_ < 0) {
    Log::Fail(FMT_STRING("Could not open file {} for writing"), fname);
  } else {
    Log::Print(FMT_STRING("Opened file to write: {}"), fname);
    Log::Debug(FMT_STRING("Handle: {}"), handle_);
  }
}

Writer::~Writer()
{
  H5Fclose(handle_);
  Log::Debug(FMT_STRING("Closed handle: {}"), handle_);
}

void Writer::writeInfo(Info const &info)
{
  hid_t info_id = InfoType();
  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  hid_t const dset = H5Dcreate(handle_, Keys::Info.c_str(), info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail(FMT_STRING("Could not create info struct in file {}, code: {}"), handle_, dset);
  }
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) {
    Log::Fail(FMT_STRING("Could not write info struct in file {}, code: {}"), handle_, status);
  }
  Log::Debug(FMT_STRING("Wrote info struct"));
}

void Writer::writeMeta(std::map<std::string, float> const &meta)
{
  Log::Print(FMT_STRING("Writing meta data"));
  auto m_group = H5Gcreate(handle_, Keys::Meta.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  herr_t status;
  for (auto const &kvp : meta) {
    hid_t const dset =
      H5Dcreate(m_group, kvp.first.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(kvp.second));
    status = H5Dclose(dset);
  }
  status = H5Sclose(space);
  if (status != 0) {
    Log::Fail(FMT_STRING("Exception occured storing meta-data in file {}"), handle_);
  }
}

void Writer::writeTrajectory(Trajectory const &t)
{
  writeInfo(t.info());
  HD5::store_tensor(handle_, Keys::Trajectory, t.points());
  HD5::store_tensor(handle_, "frames", t.frames());
}

bool Writer::exists(std::string const &name) const
{
  return HD5::Exists(handle_, name);
}

template <typename Scalar, int ND>
void Writer::writeTensor(Eigen::Tensor<Scalar, ND> const &t, std::string const &label)
{
  HD5::store_tensor(handle_, label, t);
}

template void Writer::writeTensor<float, 2>(R2 const &, std::string const &);
template void Writer::writeTensor<float, 3>(R3 const &, std::string const &);
template void Writer::writeTensor<float, 4>(R4 const &, std::string const &);
template void Writer::writeTensor<float, 5>(R5 const &, std::string const &);
template void Writer::writeTensor<Cx, 3>(Cx3 const &, std::string const &);
template void Writer::writeTensor<Cx, 4>(Cx4 const &, std::string const &);
template void Writer::writeTensor<Cx, 5>(Cx5 const &, std::string const &);
template void Writer::writeTensor<Cx, 6>(Cx6 const &, std::string const &);

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
