#include "util.hpp"

#include <itkImportImageFilter.h>

namespace merlin {
auto Import(rl::Re3Map const data, rl::Info const info) -> ImageType::Pointer
{
  using TImport = itk::ImportImageFilter<float, 3>;

  TImport::IndexType st;
  st.Fill(0);
  TImport::SizeType sz;
  std::copy_n(data.dimensions().begin(), 3, sz.begin());
  TImport::RegionType region;
  region.SetIndex(st);
  region.SetSize(sz);

  TImport::SpacingType s;
  std::copy_n(info.voxel_size.cbegin(), 3, s.begin());
  TImport::OriginType o;
  std::copy_n(info.origin.cbegin(), 3, o.begin());
  TImport::DirectionType d;
  for (Index ii = 0; ii < 3; ii++) {
    for (Index ij = 0; ij < 3; ij++) {
      d(ii, ij) = info.direction(ii, ij);
    }
  }

  auto import = TImport::New();
  import->SetRegion(region);
  import->SetSpacing(s);
  import->SetOrigin(o);
  import->SetDirection(d);
  import->SetImportPointer(data.data(), data.size(), false);
  import->Update();
  return import->GetOutput();
}

auto ITKToRIESLING(TransformType::Pointer tfm) -> rl::Transform
{
  auto const    m = tfm->GetMatrix();
  auto const    o = tfm->GetTranslation();
  rl::Transform t;
  for (Index ii = 0; ii < 3; ii++) {
    for (Index ij = 0; ij < 3; ij++) {
      t.R(ij, ii) = m(ii, ij); // Seem to need a transpose here. Not clear why.
    }
    t.δ(ii) = o[ii];
  }
  return t;
}
}
