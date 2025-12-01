#include "writer.hpp"

#include "../info.hpp"
#include "../log/log.hpp"
#include "../types.hpp"
#include <filesystem>
#include <hdf5.h>
#include <hdf5_hl.h>

namespace rl {
namespace HD5 {

namespace {
Index deflate = 2;
}

void SetDeflate(Index d) { deflate = d; }

Writer::Writer(std::string const &fname, bool const append)
{
  Init();
  auto const p = std::filesystem::path(fname).replace_extension(".h5");
  if (append) {
    if (!std::filesystem::exists(p)) { throw Log::Failure("HD5", "File does not exist: {}", p.string()); }
    handle_ = H5Fopen(p.string().c_str(), H5F_ACC_RDWR, H5P_DEFAULT);
  } else {
    handle_ = H5Fcreate(p.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  }
  if (handle_ < 0) {
    throw Log::Failure("HD5", "Could not open file {} for writing because: {}", fname, GetError());
  } else {
    Log::Print("HD5", "Opened {} for writing id {}", fname, handle_);
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
  if (status) { throw Log::Failure("HD5", "Could not write string {} into handle {}, code: {}", label, handle_, status); }
}

void Writer::writeStrings(std::string const &label, std::vector<std::string> const &strings)
{
  herr_t      status;
  hsize_t     dim[1] = {strings.size()};
  auto const  space = H5Screate_simple(1, dim, NULL);
  hid_t const tid = H5Tcopy(H5T_C_S1);
  H5Tset_size(tid, H5T_VARIABLE);
  H5Tset_cset(tid, H5T_CSET_UTF8);
  std::vector<char const *> ptrs(strings.size());
  for (size_t ii = 0; ii < strings.size(); ii++) {
    ptrs[ii] = strings[ii].c_str();
  }
  hid_t const dset = H5Dcreate(handle_, label.c_str(), tid, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  status = H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, ptrs.data());
  status = H5Dclose(dset);
  status = H5Sclose(space);
  if (status) { throw Log::Failure("HD5", "Could not write string {} into handle {}, code: {}", label, handle_, status); }
}

template <> void Writer::writeStruct(std::string const &lbl, Info const &info) const
{
  hid_t       info_id = InfoType();
  hsize_t     dims[1] = {1};
  auto const  space = H5Screate_simple(1, dims, NULL);
  hid_t const dset = H5Dcreate(handle_, lbl.c_str(), info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) { throw Log::Failure("HD5", "Could not create info struct in file {}, code: {}", handle_, dset); }
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) { throw Log::Failure("HD5", "Could not write info struct in file {}, code: {}", handle_, status); }
  Log::Debug("HD5", "Wrote info struct");
}

template <> void Writer::writeStruct(std::string const &lbl, Transform const &tfm) const
{
  hid_t       tfm_id = TransformType();
  hsize_t     dims[1] = {1};
  auto const  space = H5Screate_simple(1, dims, NULL);
  hid_t const dset = H5Dcreate(handle_, lbl.c_str(), tfm_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) { throw Log::Failure("HD5", "Could not create transform struct in file {}, code: {}", handle_, dset); }
  herr_t status;
  status = H5Dwrite(dset, tfm_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &tfm);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) { throw Log::Failure("HD5", "Could not write info struct in file {}, code: {}", handle_, status); }
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
  if (status != 0) { throw Log::Failure("HD5", "Exception occured storing meta-data in file {}", handle_); }
}

bool Writer::exists(std::string const &name) const { return HD5::Exists(handle_, name); }

template <typename Scalar, size_t N>
void Writer::writeTensor(std::string const &name, Shape<N> const &shape, Scalar const *data, DNames<N> const &labels)
{
  for (size_t ii = 0; ii < N; ii++) {
    if (shape[ii] == 0) { throw Log::Failure("HD5", "Tensor {} had a zero dimension. Dims: {}", name, shape); }
  }

  hsize_t ds_dims[N], chunk_dims[N];
  // HD5=row-major, Eigen=col-major, so need to reverse the dimensions
  std::copy_n(shape.rbegin(), N, ds_dims);
  std::copy_n(ds_dims, N, chunk_dims);
  // Try to stop chunk dimension going over 4 gig
  Index sizeInBytes = Product(shape) * sizeof(Scalar);
  Index dimToShrink = 0;
  while (sizeInBytes >= (1L << 32L)) {
    if (chunk_dims[dimToShrink] > 1) {
      chunk_dims[dimToShrink] /= 2;
      sizeInBytes /= 2;
    }
    dimToShrink = (dimToShrink + 1) % N;
  }

  auto const space = H5Screate_simple(N, ds_dims, NULL);
  auto const plist = H5Pcreate(H5P_DATASET_CREATE);
  CheckedCall(H5Pset_chunk(plist, N, chunk_dims), "setting chunk");
  if (deflate > 0) {
    CheckedCall(H5Pset_deflate(plist, deflate), "setting deflate");
  }

  hid_t const tid = type<Scalar>();
  hid_t const dset = H5Dcreate(handle_, name.c_str(), tid, space, H5P_DEFAULT, plist, H5P_DEFAULT);
  if (dset < 0) { throw Log::Failure("HD5", "Could not create dataset {} dimensions {} error {}", name, shape, GetError()); }
  auto l = labels.rbegin();
  for (size_t ii = 0; ii < N; ii++) {
    CheckedCall(H5DSset_label(dset, ii, l->c_str()), fmt::format("dataset {} dimension {} label {}", name, ii, *l));
    l++;
  }
  if (dset < 0) {
    throw Log::Failure("HD5", "Could not create tensor {}. Dims {}. Error {}", name, fmt::join(shape, ","), HD5::GetError());
  }
  CheckedCall(H5Dwrite(dset, tid, H5S_ALL, H5S_ALL, H5P_DEFAULT, data), "Writing data");
  CheckedCall(H5Pclose(plist), "closing plist");
  CheckedCall(H5Sclose(space), "closing space");
  CheckedCall(H5Dclose(dset), "closing dataset");

  Log::Debug("HD5", "Wrote tensor {}", name);
}

template void Writer::writeTensor<Index, 1>(std::string const &, Shape<1> const &, Index const *, DNames<1> const &);
template void Writer::writeTensor<float, 1>(std::string const &, Shape<1> const &, float const *, DNames<1> const &);
template void Writer::writeTensor<float, 2>(std::string const &, Shape<2> const &, float const *, DNames<2> const &);
template void Writer::writeTensor<float, 3>(std::string const &, Shape<3> const &, float const *, DNames<3> const &);
template void Writer::writeTensor<float, 4>(std::string const &, Shape<4> const &, float const *, DNames<4> const &);
template void Writer::writeTensor<float, 5>(std::string const &, Shape<5> const &, float const *, DNames<5> const &);
template void Writer::writeTensor<Cx, 2>(std::string const &, Shape<2> const &, Cx const *, DNames<2> const &);
template void Writer::writeTensor<Cx, 3>(std::string const &, Shape<3> const &, Cx const *, DNames<3> const &);
template void Writer::writeTensor<Cx, 4>(std::string const &, Shape<4> const &, Cx const *, DNames<4> const &);
template void Writer::writeTensor<Cx, 5>(std::string const &, Shape<5> const &, Cx const *, DNames<5> const &);
template void Writer::writeTensor<Cx, 6>(std::string const &, Shape<6> const &, Cx const *, DNames<6> const &);

template <size_t N> void Writer::writeAttribute(std::string const &dset, std::string const &attr, Shape<N> const &val)
{
  hsize_t const szN[1] = {N}, sz1[1] = {1};
  hid_t const   long3_id = H5Tarray_create(H5T_NATIVE_LONG, 1, szN);
  auto const    space = H5Screate_simple(1, sz1, NULL);
  auto const    attrH =
    H5Acreate_by_name(handle_, dset.c_str(), attr.c_str(), long3_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  CheckedCall(H5Awrite(attrH, long3_id, val.data()), fmt::format("writing attribute {} to {}", attr, dset));
}

template void Writer::writeAttribute<1>(std::string const &dset, std::string const &attr, Shape<1> const &val);
template void Writer::writeAttribute<2>(std::string const &dset, std::string const &attr, Shape<2> const &val);
template void Writer::writeAttribute<3>(std::string const &dset, std::string const &attr, Shape<3> const &val);

} // namespace HD5
} // namespace rl
