#include "hd5-core.hpp"

#include "../info.hpp"
#include "../log.hpp"
#include <filesystem>

#include <hdf5.h>

namespace rl {
namespace HD5 {

namespace {

struct complex_f
{
  float r; /*real part*/
  float i; /*imaginary part*/
};

struct complex_d
{
  double r; /*real part*/
  double i; /*imaginary part*/
};

hid_t complex_fid, alternate_complex_fid, complex_did, alternate_complex_did;

} // namespace

template <> hid_t type_impl(type_tag<Index>, bool const) { return H5T_NATIVE_LONG; }

template <> hid_t type_impl(type_tag<float>, bool const) { return H5T_NATIVE_FLOAT; }

template <> hid_t type_impl(type_tag<double>, bool const) { return H5T_NATIVE_DOUBLE; }

template <> hid_t type_impl(type_tag<std::complex<float>>, bool const alt)
{
  if (alt) {
    return alternate_complex_fid;
  } else {
    return complex_fid;
  }
}

template <> hid_t type_impl(type_tag<std::complex<double>>, bool const alt)
{
  if (alt) {
    return alternate_complex_did;
  } else {
    return complex_did;
  }
}

herr_t ConvertFloatComplex(hid_t, hid_t, H5T_cdata_t *, size_t n, size_t, size_t, void *buf, void *, hid_t)
{
  // HDF5 wants the conversion in place
  // Cheat heavily and convert going backwards so we don't overwrite any values
  float *src = (float *)buf;
  Cx    *tgt = (Cx *)buf;
  for (Index ii = n - 1; ii >= 0; ii--) {
    tgt[ii] = Cx(src[ii]);
  }
  return 0;
}

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
    if (err < 0) { throw Log::Failure("HD5", "Could not initialise HDF5, code: {}", err); }
    NeedsInit = false;
    hid_t fid = type_impl(type_tag<float>{});

    complex_fid = H5Tcreate(H5T_COMPOUND, sizeof(complex_f));
    CheckedCall(H5Tinsert(complex_fid, "r", HOFFSET(complex_f, r), fid), "inserting .r");
    CheckedCall(H5Tinsert(complex_fid, "i", HOFFSET(complex_f, i), fid), "inserting .i");
    H5Tregister(H5T_PERS_HARD, "real->complex", H5T_NATIVE_FLOAT, complex_fid, ConvertFloatComplex);

    alternate_complex_fid = H5Tcreate(H5T_COMPOUND, sizeof(complex_f));
    CheckedCall(H5Tinsert(alternate_complex_fid, "real", HOFFSET(complex_f, r), fid), "inserting .real ");
    CheckedCall(H5Tinsert(alternate_complex_fid, "imag", HOFFSET(complex_f, i), fid), "inserting .imag");
    H5Tregister(H5T_PERS_HARD, "real->complex", H5T_NATIVE_FLOAT, alternate_complex_fid, ConvertFloatComplex);

    hid_t did = type_impl(type_tag<double>{});

    complex_did = H5Tcreate(H5T_COMPOUND, sizeof(complex_d));
    CheckedCall(H5Tinsert(complex_did, "r", HOFFSET(complex_d, r), did), "inserting .r d");
    CheckedCall(H5Tinsert(complex_did, "i", HOFFSET(complex_d, i), did), "inserting .i d");

    alternate_complex_did = H5Tcreate(H5T_COMPOUND, sizeof(complex_d));
    CheckedCall(H5Tinsert(alternate_complex_did, "real", HOFFSET(complex_d, r), did), "inserting .real d");
    CheckedCall(H5Tinsert(alternate_complex_did, "imag", HOFFSET(complex_d, i), did), "inserting .imag d");

    Log::Debug("HD5", "Initialised HDF5");
  } else {
    Log::Debug("HD5", "Already initialised");
  }
}

// Saves the error at the top (bottom) of the stack in the supplied string
herr_t ErrorWalker(unsigned n, const H5E_error2_t *err_desc, void *data)
{
  std::string *str = (std::string *)data;
  if (n == 0) { *str = fmt::format("{}\n", err_desc->desc); }
  return 0;
}

std::string GetError()
{
  std::string error_string;
  H5Ewalk(H5Eget_current_stack(), H5E_WALK_UPWARD, &ErrorWalker, (void *)&error_string);
  return error_string;
}

void CheckedCall(herr_t status, std::string const &msg)
{
  if (status) { throw Log::Failure("HD5", "Error {}. Status {}. Error: {}\n", msg, status, GetError()); }
}

hid_t InfoType()
{
  hid_t   info_id = H5Tcreate(H5T_COMPOUND, sizeof(Info));
  hsize_t sz3[1] = {3};
  hid_t   float3_id = H5Tarray_create(H5T_NATIVE_FLOAT, 1, sz3);
  hsize_t sz33[2] = {3, 3};
  hid_t   float33_id = H5Tarray_create(H5T_NATIVE_FLOAT, 2, sz33);
  CheckedCall(H5Tinsert(info_id, "voxel_size", HOFFSET(Info, voxel_size), float3_id), "inserting voxel size field");
  CheckedCall(H5Tinsert(info_id, "origin", HOFFSET(Info, origin), float3_id), "inserting oring field");
  CheckedCall(H5Tinsert(info_id, "direction", HOFFSET(Info, direction), float33_id), "inserting direction field");
  CheckedCall(H5Tinsert(info_id, "tr", HOFFSET(Info, tr), H5T_NATIVE_FLOAT), "inserting tr field");
  return info_id;
}

void CheckInfoType(hid_t handle)
{
  // Hard code for now until the fields in InfoType are replaced with some kind of auto-gen
  // Also use vector instead of array so I don't forget to change the size if the members change
  std::vector<std::string> const names{"voxel_size", "origin", "direction", "tr"};

  if (handle < 0) { throw Log::Failure("HD5", "Info struct does not exist"); }
  auto const dtype = H5Dget_type(handle);
  size_t     n_members = H5Tget_nmembers(dtype);
  // Re-ordered and extra fields are okay. Missing is not
  for (auto const &check_name : names) {
    bool found = false;
    for (size_t ii = 0; ii < n_members; ii++) {
      std::string const member_name(H5Tget_member_name(dtype, ii));
      if (member_name == check_name) {
        found = true;
        break;
      }
    }
    if (!found) { throw Log::Failure("HD5", "Field {} not found in header info", check_name); }
  }
}

auto Exists(hid_t const parent, std::string const &name) -> bool { return (H5Lexists(parent, name.c_str(), H5P_DEFAULT) > 0); }

herr_t AddName(hid_t, const char *name, const H5L_info_t *, void *opdata)
{
  auto names = reinterpret_cast<std::vector<std::string> *>(opdata);
  names->push_back(name);
  return 0;
}

std::vector<std::string> List(Handle h)
{
  std::vector<std::string> names;
  H5Literate(h, H5_INDEX_NAME, H5_ITER_INC, NULL, AddName, &names);

  std::erase_if(names, [h](std::string const &name) {
    H5O_info_t info;
    H5Oget_info_by_name(h, name.c_str(), &info, H5O_INFO_BASIC, H5P_DEFAULT);
    return info.type != H5O_TYPE_DATASET;
  });

  return names;
}

} // namespace HD5
} // namespace rl
