#include "hd5.h"
#include "hd5.hpp"

namespace HD5 {

void init()
{
  auto err = H5open();
  err = H5Eset_auto(H5E_DEFAULT, nullptr, nullptr); // Turn off HD5 error messages
  if (err < 0) {
    throw std::runtime_error("Could not initialise HDF5");
  }
}

Handle open_file(std::string const &path, Mode const mode)
{
  switch (mode) {
  case Mode::ReadOnly:
    return H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  case Mode::WriteOnly:
    return H5Fcreate(path.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  default:
    return -1;
  }
}

void close_file(Handle const &file)
{
  H5Fclose(file);
}

std::string get_name(Handle const &h)
{
  auto const sz = H5Iget_name(h, NULL, 0);
  std::string name(sz, 0);
  H5Iget_name(h, name.data(), sz + 1);
  return name;
}

Handle create_group(Handle const &parent, std::string const &name)
{
  return H5Gcreate(parent, name.c_str(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
}

Handle open_group(Handle const &parent, std::string const &name)
{
  return H5Gopen(parent, name.c_str(), H5P_DEFAULT);
}

void close_group(Handle const &group)
{
  H5Gclose(group);
}

herr_t h5_names(hid_t id, const char *name, const H5L_info_t *linfo, void *opdata)
{
  auto names = reinterpret_cast<std::vector<std::string> *>(opdata);
  names->push_back(name);
  return 0;
}

std::vector<std::string> list_group(Handle const &group)
{
  std::vector<std::string> names;
  H5Literate(group, H5_INDEX_NAME, H5_ITER_INC, NULL, h5_names, &names);
  return names;
}

Handle open_data(Handle const &parent, std::string const &name)
{
  auto const dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
  if (dset < 0) {
    throw std::runtime_error(fmt::format("Could not open dataset: {}", name));
  }
  return dset;
}

hid_t info_type()
{
  hid_t info_id = H5Tcreate(H5T_COMPOUND, sizeof(RadialInfo));
  hsize_t sz3[1] = {3};
  hid_t long3_id = H5Tarray_create(H5T_NATIVE_LONG, 1, sz3);
  hid_t float3_id = H5Tarray_create(H5T_NATIVE_FLOAT, 1, sz3);
  hsize_t sz9[1] = {9};
  hid_t float9_id = H5Tarray_create(H5T_NATIVE_FLOAT, 1, sz9);
  herr_t status;
  status = H5Tinsert(info_id, "matrix", HOFFSET(RadialInfo, matrix), long3_id);
  status = H5Tinsert(info_id, "voxel_size", HOFFSET(RadialInfo, voxel_size), float3_id);
  status = H5Tinsert(info_id, "read_points", HOFFSET(RadialInfo, read_points), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "read_gap", HOFFSET(RadialInfo, read_gap), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "spokes_hi", HOFFSET(RadialInfo, spokes_hi), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "spokes_lo", HOFFSET(RadialInfo, spokes_lo), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "lo_scale", HOFFSET(RadialInfo, lo_scale), H5T_NATIVE_FLOAT);
  status = H5Tinsert(info_id, "channels", HOFFSET(RadialInfo, channels), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "volumes", HOFFSET(RadialInfo, volumes), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "tr", HOFFSET(RadialInfo, tr), H5T_NATIVE_FLOAT);
  status = H5Tinsert(info_id, "origin", HOFFSET(RadialInfo, origin), float3_id);
  status = H5Tinsert(info_id, "direction", HOFFSET(RadialInfo, direction), float9_id);
  if (status) {
    throw std::runtime_error("Exception occurred creating radial info type");
  }
  return info_id;
}

void store_info(Handle const &parent, RadialInfo const &info)
{
  hid_t info_id = info_type();
  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  hid_t const dset =
      H5Dcreate(parent, "info", info_id, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  herr_t status;
  status = H5Dwrite(dset, info_id, H5S_ALL, H5S_ALL, H5P_DEFAULT, &info);
  status = H5Sclose(space);
  status = H5Dclose(dset);
  if (status != 0) {
    throw std::runtime_error("Exception occurred storing radial info struct");
  }
}

void store_map(Handle const &parent, std::map<std::string, float> const &meta)
{
  hsize_t dims[1] = {1};
  auto const space = H5Screate_simple(1, dims, NULL);
  herr_t status;
  for (auto const &kvp : meta) {
    hid_t const dset = H5Dcreate(
        parent, kvp.first.c_str(), H5T_NATIVE_FLOAT, space, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    status = H5Dwrite(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(kvp.second));
    status = H5Dclose(dset);
  }
  status = H5Sclose(space);
  if (status != 0) {
    throw std::runtime_error("Exception occured storing meta-data");
  }
}

void load_info(Handle const &parent, RadialInfo &info)
{
  hid_t const info_id = info_type();
  hid_t const dset = H5Dopen(parent, "info", H5P_DEFAULT);
  hid_t const space = H5Dget_space(dset);
  herr_t status = H5Dread(dset, info_id, space, H5S_ALL, H5P_DATASET_XFER_DEFAULT, &info);
  status = H5Dclose(dset);
  if (status != 0) {
    throw std::runtime_error("Could not load radial info struct");
  }
}

void load_map(Handle const &parent, std::map<std::string, float> &meta)
{
  auto names = list_group(parent);
  meta.clear();
  herr_t status = 0;
  for (auto const &name : names) {
    hid_t const dset = H5Dopen(parent, name.c_str(), H5P_DEFAULT);
    float value;
    status = H5Dread(dset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &value);
    status = H5Dclose(dset);
    meta[name] = value;
  }
  if (status != 0) {
    throw std::runtime_error("Could not load meta-data");
  }
}

template void store_tensor(Handle const &parent, std::string const &name, R3 const &data);
template void store_tensor(Handle const &parent, std::string const &name, Cx3 const &data);
template void load_tensor(Handle const &parent, std::string const &name, R3 &tensor);
template void load_tensor(Handle const &parent, std::string const &name, Cx3 &tensor);
template void store_array(Handle const &parent, std::string const &name, Eigen::ArrayXf const &t);
template void load_array(Handle const &parent, std::string const &name, Eigen::ArrayXf &t);
template void
store_array(Handle const &parent, std::string const &name, Eigen::Array<float, 3, -1> const &a);
template void
load_array(Handle const &parent, std::string const &name, Eigen::Array<float, 3L, -1L> &a);

} // namespace HD5