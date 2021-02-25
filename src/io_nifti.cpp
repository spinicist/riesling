#include "io_nifti.h"
#include "itkImage.h"
#include "itkImageFileWriter.h"
#include "itkImportImageFilter.h"
#include <fmt/ostream.h>

template <typename T, int ND>
void WriteNifti(
    Info const &info, Eigen::Tensor<T, ND> const &img, std::string const &fname, Log const &log)
{
  using Image = itk::Image<T, ND>;
  using Importer = itk::ImportImageFilter<T, ND>;
  using Writer = itk::ImageFileWriter<Image>;

  typename Importer::Pointer import = Importer::New();
  typename Importer::SizeType sz;
  typename Importer::IndexType st;
  auto const dims = img.dimensions();
  std::copy_n(dims.begin(), ND, &sz[0]);
  std::fill_n(&st[0], ND, 0.);
  typename Importer::RegionType const region{st, sz};
  import->SetRegion(region);

  double spacing[ND];
  std::copy_n(info.voxel_size.begin(), 3, spacing);
  if constexpr (ND == 4) {
    spacing[3] = info.tr;
  }
  import->SetSpacing(spacing);

  double origin[ND];
  std::copy_n(info.origin.begin(), 3, origin);
  if constexpr (ND == 4) {
    origin[3] = 0;
  }
  import->SetOrigin(origin);

  itk::Matrix<double, ND, ND> direction;
  direction.Fill(0.);
  // This appears to need a transpose. I don't know why
  for (auto ir = 0; ir < 3; ir++) {
    for (auto ic = 0; ic < 3; ic++) {
      direction(ic, ir) = info.direction(ir, ic);
    }
  }
  if constexpr (ND == 4) {
    direction(3, 3) = 1.;
  }
  import->SetDirection(direction);

  auto const n_vox = img.size();
  import->SetImportPointer(const_cast<T *>(img.data()), n_vox, false);
  typename Writer::Pointer write = Writer::New();
  write->SetFileName(fname);
  write->SetInput(import->GetOutput());
  log.info(FMT_STRING("Writing file: {}"), fname);
  write->Update();
}

template void WriteNifti(Info const &, Cx3 const &, std::string const &, Log const &);
template void WriteNifti(Info const &, Cx4 const &, std::string const &, Log const &);
template void WriteNifti(Info const &, R3 const &, std::string const &, Log const &);
template void WriteNifti(Info const &, R4 const &, std::string const &, Log const &);

template <typename T>
void WriteNifti(Eigen::Matrix<T, -1, -1> const &m, std::string const &fname, Log const &log)
{
  using Image = itk::Image<T, 2>;
  using Importer = itk::ImportImageFilter<T, 2>;
  using Writer = itk::ImageFileWriter<Image>;

  typename Importer::Pointer import = Importer::New();
  typename Importer::SizeType sz{static_cast<size_t>(m.rows()), static_cast<size_t>(m.cols())};
  typename Importer::IndexType st{0, 0};
  typename Importer::RegionType const region{st, sz};
  import->SetRegion(region);
  double origin[2] = {0., 0.};
  import->SetOrigin(origin);
  double spacing[2] = {1., 1.};
  import->SetSpacing(spacing);
  auto const n_vox = m.size();
  import->SetImportPointer(const_cast<T *>(m.data()), n_vox, false);
  typename Writer::Pointer write = Writer::New();
  write->SetFileName(fname);
  write->SetInput(import->GetOutput());
  log.info(FMT_STRING("Writing file: {}"), fname);
  write->Update();
}

template void WriteNifti(Eigen::MatrixXcf const &m, std::string const &fname, Log const &log);