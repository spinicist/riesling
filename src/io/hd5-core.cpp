#include "hd5-core.hpp"
#include "info.h"
#include "log.h"
#include <filesystem>

namespace rl {
namespace HD5 {

void Init()
{
  static bool NeedsInit = true;

  if (NeedsInit) {
    auto err = H5open();
    // herr_t (*old_func)(Index Index, void *);
    // void *old_client_data;
    // hid_t errorStack;
    // err = H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
    err = H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
    if (err < 0) {
      Log::Fail(FMT_STRING("Could not initialise HDF5, code: {}"), err);
    }
    NeedsInit = false;
    Log::Debug(FMT_STRING("Initialised HDF5"));
  } else {
    Log::Debug(FMT_STRING("HDF5 already initialised"));
  }
}

// Saves the error at the top (bottom) of the stack in the supplied string
herr_t ErrorWalker(unsigned n, const H5E_error2_t *err_desc, void *data)
{
  std::string *str = (std::string *)data;
  if (n == 0) {
    *str = fmt::format(FMT_STRING("{}\n"), err_desc->desc);
  }
  return 0;
}

std::string GetError()
{
  std::string error_string;
  H5Ewalk(H5Eget_current_stack(), H5E_WALK_UPWARD, &ErrorWalker, (void *)&error_string);
  return error_string;
}

hid_t InfoType()
{
  hid_t info_id = H5Tcreate(H5T_COMPOUND, sizeof(Info));
  hsize_t sz3[1] = {3};
  hid_t long3_id = H5Tarray_create(H5T_NATIVE_LONG, 1, sz3);
  hid_t float3_id = H5Tarray_create(H5T_NATIVE_FLOAT, 1, sz3);
  hsize_t sz9[1] = {9};
  hid_t float9_id = H5Tarray_create(H5T_NATIVE_FLOAT, 1, sz9);
  herr_t status;
  status = H5Tinsert(info_id, "matrix", HOFFSET(Info, matrix), long3_id);
  status = H5Tinsert(info_id, "voxel_size", HOFFSET(Info, voxel_size), float3_id);
  status = H5Tinsert(info_id, "read_points", HOFFSET(Info, read_points), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "spokes", HOFFSET(Info, spokes), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "channels", HOFFSET(Info, channels), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "type", HOFFSET(Info, type), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "volumes", HOFFSET(Info, volumes), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "frames", HOFFSET(Info, frames), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "tr", HOFFSET(Info, tr), H5T_NATIVE_FLOAT);
  status = H5Tinsert(info_id, "origin", HOFFSET(Info, origin), float3_id);
  status = H5Tinsert(info_id, "direction", HOFFSET(Info, direction), float9_id);
  if (status) {
    Log::Fail(FMT_STRING("Could not create Info struct type in HDF5, code: {}"), status);
  }
  return info_id;
}

void CheckInfoType(hid_t handle)
{
  // Hard code for now until the fields in InfoType are replaced with some kind of auto-gen
  // Also use vector instead of array so I don't forget to change the size if the members change
  std::vector<std::string> const names{
    "matrix",
    "voxel_size",
    "read_points",
    "spokes",
    "channels",
    "type",
    "volumes",
    "frames",
    "tr",
    "origin",
    "direction"};

  if (handle < 0) {
    Log::Fail("Info struct does not exist");
  }
  auto const dtype = H5Dget_type(handle);
  size_t n_members = H5Tget_nmembers(dtype);
  if (n_members < names.size()) {
    Log::Fail(FMT_STRING("Header info had {} members, should be {}"), n_members, names.size());
  }
  // Re-ordered fields are okay. Missing is not
  for (auto const &check_name : names) {
    bool found = false;
    for (size_t ii = 0; ii < n_members; ii++) {
      std::string const member_name(H5Tget_member_name(dtype, ii));
      if (member_name == check_name) {
        found = true;
        break;
      }
    }
    if (!found) {
      Log::Fail(FMT_STRING("Field {} not found in header info"), check_name);
    }
  }
}

bool Exists(hid_t const parent, std::string const name)
{
  return (H5Lexists(parent, name.c_str(), H5P_DEFAULT) > 0);
}

template <>
hid_t type_impl(type_tag<Index>)
{
  return H5T_NATIVE_LONG;
}

template <>
hid_t type_impl(type_tag<float>)
{
  return H5T_NATIVE_FLOAT;
}

template <>
hid_t type_impl(type_tag<double>)
{
  return H5T_NATIVE_DOUBLE;
}

template <>
hid_t type_impl(type_tag<std::complex<float>>)
{
  struct complex_t
  {
    float r; /*real part*/
    float i; /*imaginary part*/
  };

  hid_t scalar_id = type_impl(type_tag<float>{});
  hid_t complex_id = H5Tcreate(H5T_COMPOUND, sizeof(complex_t));
  herr_t status;
  status = H5Tinsert(complex_id, "r", HOFFSET(complex_t, r), scalar_id);
  status = H5Tinsert(complex_id, "i", HOFFSET(complex_t, i), scalar_id);
  if (status) {
    throw std::runtime_error("Exception occurred creating complex datatype " + std::to_string(status));
  }
  return complex_id;
}

herr_t AddName(hid_t id, const char *name, const H5L_info_t *linfo, void *opdata)
{
  auto names = reinterpret_cast<std::vector<std::string> *>(opdata);
  names->push_back(name);
  return 0;
}

std::vector<std::string> List(Handle h)
{
  std::vector<std::string> names;
  H5Literate(h, H5_INDEX_NAME, H5_ITER_INC, NULL, AddName, &names);
  return names;
}

} // namespace HD5
} // namespace rl
