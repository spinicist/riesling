#include "io_hd5.h"
#include "info.h"
#include "io_hd5.hpp"
#include <Eigen/Eigenvalues>
#include <filesystem>
#include <fmt/format.h>

namespace HD5 {

void Init()
{
  static bool NeedsInit = true;

  if (NeedsInit) {

    auto err = H5open();
    // herr_t (*old_func)(long long, void *);
    // void *old_client_data;
    // hid_t errorStack;
    // err = H5Eget_auto(H5E_DEFAULT, &old_func, &old_client_data);
    err = H5Eset_auto(H5E_DEFAULT, nullptr, nullptr);
    if (err < 0) {
      Log::Fail("Could not initialise HDF5, code: {}", err);
    }
    NeedsInit = false;
  }
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
  status = H5Tinsert(info_id, "read_gap", HOFFSET(Info, read_gap), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "spokes_hi", HOFFSET(Info, spokes_hi), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "spokes_lo", HOFFSET(Info, spokes_lo), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "lo_scale", HOFFSET(Info, lo_scale), H5T_NATIVE_FLOAT);
  status = H5Tinsert(info_id, "channels", HOFFSET(Info, channels), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "type", HOFFSET(Info, type), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "volumes", HOFFSET(Info, volumes), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "echoes", HOFFSET(Info, echoes), H5T_NATIVE_LONG);
  status = H5Tinsert(info_id, "tr", HOFFSET(Info, tr), H5T_NATIVE_FLOAT);
  status = H5Tinsert(info_id, "origin", HOFFSET(Info, origin), float3_id);
  status = H5Tinsert(info_id, "direction", HOFFSET(Info, direction), float9_id);
  if (status) {
    Log::Fail("Could not create Info struct type in HDF5, code: {}", status);
  }
  return info_id;
}

void CheckInfoType(hid_t handle)
{
  // Hard code for now until the fields in InfoType are replaced with some kind of auto-gen
  constexpr int N = 14;
  std::array<std::string, N> const names{
    "matrix",
    "voxel_size",
    "read_points",
    "read_gap",
    "spokes_hi",
    "spokes_lo",
    "lo_scale",
    "channels",
    "type",
    "volumes",
    "echoes",
    "tr",
    "origin",
    "direction"};

  auto const dtype = H5Dget_type(handle);
  int n_members = H5Tget_nmembers(dtype);
  if (n_members != N) {
    Log::Fail("Header info had {} members, should be {}", n_members, N);
  }
  // Re-orderd fields are okay. Missing is not
  for (auto const &check_name : names) {
    bool found = false;
    for (int ii = 0; ii < N; ii++) {
      std::string const member_name(H5Tget_member_name(dtype, ii));
      if (member_name == check_name) {
        found = true;
        break;
      }
    }
    if (!found) {
      Log::Fail("Field {} not found in header info", check_name);
    }
  }
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
    throw std::runtime_error(
      "Exception occurred creating complex datatype " + std::to_string(status));
  }
  return complex_id;
}

} // namespace HD5
