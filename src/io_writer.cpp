#include "io_writer.h"
#include "io_hd5.h"
#include "io_hd5.hpp"

namespace HD5 {

void store_matrix(
  Handle const &parent,
  std::string const &name,
  Eigen::Ref<Eigen::MatrixXf const> const &data,
  Log const &log)
{
  herr_t status;
  hsize_t ds_dims[2], chunk_dims[2];
  // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
  ds_dims[0] = data.cols();
  ds_dims[1] = data.rows();
  std::copy_n(ds_dims, 2, chunk_dims);
  auto const space = H5Screate_simple(2, ds_dims, NULL);
  auto const plist = H5Pcreate(H5P_DATASET_CREATE);
  status = H5Pset_deflate(plist, 2);
  status = H5Pset_chunk(plist, 2, chunk_dims);

  hid_t const tid = type<float>();
  hid_t const dset = H5Dcreate(parent, name.c_str(), tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data.data());
  status = H5Pclose(plist);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status) {
    Log::Fail("Could not write tensor {}, code: {}", name, status);
  } else {
    log.info("Wrote dataset: {}", name);
  }
}

Writer::Writer(std::string const &fname, Log &log)
  : log_(log)
{
  Init();
  handle_ = H5Fcreate(fname.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  if (handle_ < 0) {
    Log::Fail(FMT_STRING("Could not open file {} for writing"), fname);
  } else {
    log_.info(FMT_STRING("Opened file {} for writing data"), fname);
  }
}

Writer::~Writer()
{
  H5Fclose(handle_);
}

void Writer::writeInfo(Info const &info)
{
  log_.info("Writing info struct");
  hid_t info_id = InfoType();
  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  hid_t const dset =
    H5Dcreate(handle_, Keys::Info.c_str(), info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) {
    Log::Fail("Could not create info struct, code: {}", dset);
  }
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) {
    Log::Fail("Could not write Info struct, code: {}", status);
  }
}

void Writer::writeMeta(std::map<std::string, float> const &meta)
{
  log_.info("Writing meta data");
  auto m_group = H5Gcreate(handle_, Keys::Meta.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  herr_t status;
  for (auto const &kvp : meta) {
    hid_t const dset = H5Dcreate(
      m_group, kvp.first.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(kvp.second));
    status = H5Dclose(dset);
  }
  status = H5Sclose(space);
  if (status != 0) {
    throw std::runtime_error("Exception occured storing meta-data");
  }
}

void Writer::writeTrajectory(Trajectory const &t)
{
  writeInfo(t.info());
  HD5::store_tensor(handle_, Keys::Trajectory, t.points(), log_);
}

void Writer::writeSDC(R2 const &sdc)
{
  HD5::store_tensor(handle_, Keys::SDC, sdc, log_);
}

void Writer::writeSENSE(Cx4 const &s)
{
  HD5::store_tensor(handle_, Keys::SENSE, s, log_);
}

void Writer::writeNoncartesian(Cx4 const &t)
{
  HD5::store_tensor(handle_, Keys::Noncartesian, t, log_);
}

void Writer::writeCartesian(Cx4 const &t)
{
  HD5::store_tensor(handle_, Keys::Cartesian, t, log_);
}

void Writer::writeImage(R4 const &t)
{
  HD5::store_tensor(handle_, Keys::Image, t, log_);
}

void Writer::writeImage(Cx4 const &t)
{
  HD5::store_tensor(handle_, Keys::Image, t, log_);
}

void Writer::writeBasis(R2 const &b)
{
  HD5::store_tensor(handle_, Keys::Basis, b, log_);
}

void Writer::writeDynamics(R2 const &d)
{
  HD5::store_tensor(handle_, Keys::Dynamics, d, log_);
}

template <typename Scalar, int ND>
void Writer::writeTensor(Eigen::Tensor<Scalar, ND> const &t, std::string const &label)
{
  HD5::store_tensor(handle_, label, t, log_);
}

template void Writer::writeTensor<float, 3>(R3 const &, std::string const &);
template void Writer::writeTensor<float, 5>(R5 const &, std::string const &);
template void Writer::writeTensor<Cx, 3>(Cx3 const &, std::string const &);
template void Writer::writeTensor<Cx, 4>(Cx4 const &, std::string const &);
template void Writer::writeTensor<Cx, 5>(Cx5 const &, std::string const &);

void Writer::writeMatrix(Eigen::Ref<Eigen::MatrixXf const> const &m, std::string const &label)
{
  HD5::store_matrix(handle_, label, m, log_);
}

void Writer::writeBasisImages(Cx5 const &bi)
{
  HD5::store_tensor(handle_, Keys::BasisImages, bi, log_);
}

} // namespace HD5
