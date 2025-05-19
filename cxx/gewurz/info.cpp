#include "info.hpp"

#include "rl/io/hd5.hpp"
#include "rl/log/log.hpp"
#include <hdf5.h>

auto InfoType() -> hid_t
{
  hid_t   info_id = H5Tcreate(H5T_COMPOUND, sizeof(gw::Info));
  hsize_t sz3[1] = {3};
  hid_t   float3_id = H5Tarray_create(H5T_NATIVE_FLOAT, 1, sz3);
  hsize_t sz33[2] = {3, 3};
  hid_t   float33_id = H5Tarray_create(H5T_NATIVE_FLOAT, 2, sz33);
  rl::HD5::CheckedCall(H5Tinsert(info_id, "voxel_size", HOFFSET(gw::Info, voxel_size), float3_id),
                       "inserting voxel size field");
  rl::HD5::CheckedCall(H5Tinsert(info_id, "origin", HOFFSET(gw::Info, origin), float3_id), "inserting oring field");
  rl::HD5::CheckedCall(H5Tinsert(info_id, "direction", HOFFSET(gw::Info, direction), float33_id), "inserting direction field");
  rl::HD5::CheckedCall(H5Tinsert(info_id, "tr", HOFFSET(gw::Info, tr), H5T_NATIVE_FLOAT), "inserting tr field");
  return info_id;
}

template <> auto rl::HD5::Reader::readStruct<gw::Info>(std::string const &id) const -> gw::Info
{
  // First get the Info struct
  hid_t const info_id = InfoType();
  hid_t const dset = H5Dopen(handle_, id.c_str(), H5P_DEFAULT);
  hid_t const space = H5Dget_space(dset);
  gw::Info    info;
  rl::HD5::CheckedCall(H5Dread(dset, info_id, space, H5S_ALL, H5P_DATASET_XFER_DEFAULT, &info), "Could not read info struct");
  rl::HD5::CheckedCall(H5Dclose(dset), "Could not close info dataset");
  return info;
}

template <> void rl::HD5::Writer::writeStruct(std::string const &lbl, gw::Info const &info) const
{
  hid_t       info_id = InfoType();
  hsize_t     dims[1] = {1};
  auto const  space = H5Screate_simple(1, dims, NULL);
  hid_t const dset = H5Dcreate(handle_, lbl.c_str(), info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  if (dset < 0) { throw rl::Log::Failure("HD5", "Could not create info struct in file {}, code: {}", handle_, dset); }
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) { throw rl::Log::Failure("HD5", "Could not write info struct in file {}, code: {}", handle_, status); }
  Log::Debug("HD5", "Wrote info struct");
}
