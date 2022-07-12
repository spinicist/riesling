#include "io/nifti.hpp"
#include "nifti1_io.h"
#include <fmt/ostream.h>

namespace {
template <typename T>
struct nifti_traits;

template <>
struct nifti_traits<float>
{
  constexpr static int dtype()
  {
    return NIFTI_TYPE_FLOAT32;
  }
  constexpr static int bytesper()
  {
    return 4;
  }
};

template <>
struct nifti_traits<std::complex<float>>
{
  constexpr static int dtype()
  {
    return NIFTI_TYPE_COMPLEX64;
  }
  constexpr static int bytesper()
  {
    return 8;
  }
};
} // namespace

namespace rl {

template <typename T, int ND>
void WriteNifti(rl::Info const &info, Eigen::Tensor<T, ND> const &img, std::string const &fname)
{
  nifti_image *ptr = nifti_simple_init_nim();
  ptr->fname = nifti_makehdrname(fname.c_str(), NIFTI_FTYPE_NIFTI1_1, false, false);
  ptr->iname = nifti_makeimgname(fname.c_str(), NIFTI_FTYPE_NIFTI1_1, false, false);
  ptr->nvox = img.size();
  ptr->xyz_units = NIFTI_UNITS_MM | NIFTI_UNITS_SEC;
  ptr->dim[0] = ptr->ndim = 4;
  ptr->dim[1] = ptr->nx = img.dimension(0);
  ptr->dim[2] = ptr->ny = img.dimension(1);
  ptr->dim[3] = ptr->nz = img.dimension(2);
  if constexpr (ND == 4) {
    ptr->dim[4] = ptr->nt = img.dimension(3);
  } else {
    ptr->dim[4] = ptr->nt = 1;
  }
  ptr->dim[5] = ptr->nu = 1;
  ptr->dim[6] = ptr->nv = 1;
  ptr->dim[7] = ptr->nw = 1;

  ptr->datatype = nifti_traits<T>::dtype();
  ptr->nbyper = nifti_traits<T>::bytesper();
  ptr->qform_code = NIFTI_XFORM_SCANNER_ANAT;
  ptr->sform_code = NIFTI_XFORM_SCANNER_ANAT;

  ptr->pixdim[1] = info.voxel_size[0];
  ptr->pixdim[2] = info.voxel_size[1];
  ptr->pixdim[3] = info.voxel_size[2];
  ptr->pixdim[4] = info.tr;

  mat44 matrix = nifti_make_orthog_mat44(
    info.direction(0, 0),
    info.direction(0, 1),
    info.direction(0, 2),
    info.direction(1, 0),
    info.direction(1, 1),
    info.direction(1, 2),
    info.direction(2, 0),
    info.direction(2, 1),
    info.direction(2, 2));
  matrix.m[0][3] = -info.origin(0);
  matrix.m[1][3] = -info.origin(1);
  matrix.m[2][3] = -info.origin(2);

  nifti_mat44_to_quatern(
    matrix,
    &(ptr->quatern_b),
    &(ptr->quatern_c),
    &(ptr->quatern_d),
    &(ptr->qoffset_x),
    &(ptr->qoffset_y),
    &(ptr->qoffset_z),
    nullptr,
    nullptr,
    nullptr,
    &(ptr->qfac));

  ptr->qto_xyz = matrix;
  ptr->sto_xyz = matrix;
  for (auto ii = 0; ii < 3; ii++) {
    for (auto jj = 0; jj < 3; jj++) {
      ptr->sto_xyz.m[ii][jj] = info.voxel_size(0) * ptr->sto_xyz.m[ii][jj];
    }
  }
  ptr->qto_ijk = nifti_mat44_inverse(ptr->qto_xyz);
  ptr->sto_ijk = nifti_mat44_inverse(ptr->sto_xyz);
  ptr->pixdim[0] = ptr->qfac;

  ptr->data = const_cast<T *>(img.data()); // To avoid copying the buffer
  Log::Print(FMT_STRING("Writing file: {}"), fname);
  nifti_image_write(ptr);
}

template void WriteNifti(rl::Info const &, Cx3 const &, std::string const &);
template void WriteNifti(rl::Info const &, Cx4 const &, std::string const &);
template void WriteNifti(rl::Info const &, R3 const &, std::string const &);
template void WriteNifti(rl::Info const &, R4 const &, std::string const &);

} // namespace rl
