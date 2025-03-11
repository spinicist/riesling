#include "args.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"

#include <itkImageRegistrationMethod.h>
#include <itkImportImageFilter.h>
#include <itkLinearInterpolateImageFunction.h>
#include <itkMeanSquaresImageToImageMetric.h>
#include <itkRegularStepGradientDescentOptimizer.h>
#include <itkVersorRigid3DTransform.h>

extern args::Group    global_group;
extern args::HelpFlag help;
extern args::Flag     verbose;

using namespace rl;

using TTfm = itk::VersorRigid3DTransform<double>;
using TImage = itk::Image<float, 3>;
using TOpt = itk::RegularStepGradientDescentOptimizer;
using TMetric = itk::MeanSquaresImageToImageMetric<TImage, TImage>;
using TInterp = itk::LinearInterpolateImageFunction<TImage, double>;
using TReg = itk::ImageRegistrationMethod<TImage, TImage>;

auto Import(Re3CMap const data, Info const info) -> TImage::Pointer
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
  import->Update();
  return import->GetOutput();
}

auto Register(TImage::Pointer fixed, TImage::Pointer moving) -> Transform
{
  auto metric = TMetric::New();
  auto transform = TTfm::New();
  auto optimizer = TOpt::New();
  auto interpolator = TInterp::New();
  auto registration = TReg::New();

  registration->SetMetric(metric);
  registration->SetOptimizer(optimizer);
  registration->SetTransform(transform);
  registration->SetInterpolator(interpolator);
  registration->SetFixedImage(fixed);
  registration->SetMovingImage(moving);
  registration->SetFixedImageRegion(fixed->GetLargestPossibleRegion());

  //  Initialize the transform
  using TPar = TReg::ParametersType;
  TPar initialParameters(transform->GetNumberOfParameters());
  initialParameters.Fill(0);
  initialParameters[5] = 1.0; // Axis of versor
  registration->SetInitialTransformParameters(initialParameters);
  optimizer->SetMaximumStepLength(4.00);
  optimizer->SetMinimumStepLength(0.01);
  optimizer->SetNumberOfIterations(200);

  // Connect an observer
  // auto observer = CommandIterationUpdate::New();
  // optimizer->AddObserver( itk::IterationEvent(), observer );

  try {
    registration->Update();
  } catch (const itk::ExceptionObject &err) {
    throw Log::Failure("Reg", "{}", err.what());
  }

  auto final = TTfm::New();
  final->SetParameters(registration->GetLastTransformParameters());
  auto const m = final->GetMatrix();
  auto const o = final->GetOffset();
  Transform t;
  for (Index ii = 0; ii < 3; ii++) {
    for (Index ij = 0; ij < 3; ij++) {
      t.R(ii, ij) = m(ii, ij);
    }
    t.δ(ii) = o[ii];
  }
  return t;
}

int main(int const argc, char const *const argv[])
{
  args::ArgumentParser          parser("MERLIN");
  args::GlobalOptions           globals(parser, global_group);
  args::Positional<std::string> iname(parser, "INPUT", "ifile HD5 file");
  args::Positional<std::string> oname(parser, "OUTPUT", "Output HD5 file");
  args::ValueFlag<std::string>  mname(parser, "MASK", "Mask HD5 file", {'m', "mask"});

  try {
    parser.ParseCLI(argc, argv);
    if (!iname) { throw args::Error("No input file specified"); }
    if (!oname) { throw args::Error("No output file specified"); }
    HD5::Reader ifile(iname.Get());
    HD5::Writer ofile(oname.Get());
    Re4 const   idata = ifile.readTensor<Cx5>().abs().chip<4>(0);
    auto const  info = ifile.readInfo();
    auto const  fixed = Import(CChipMap(idata, 0), info);
    for (Index ii = 1; ii < idata.dimension(3); ii++) {
      auto const moving = Import(CChipMap(idata, ii), info);
      ofile.writeTransform(Register(fixed, moving), fmt::format("{:02d}", ii));
    }
    Log::End();
  } catch (args::Help &) {
    fmt::print(stderr, "{}\n", parser.Help());
    return EXIT_SUCCESS;
  } catch (args::Error &e) {
    fmt::print(stderr, "{}\n", parser.Help());
    fmt::print(stderr, fmt::fg(fmt::terminal_color::bright_red), "{}\n", e.what());
    return EXIT_FAILURE;
  } catch (Log::Failure &f) {
    Log::Fail(f);
    Log::End();
    return EXIT_FAILURE;
  } catch (std::exception const &e) {
    Log::Fail(Log::Failure("None", "{}", e.what()));
    Log::End();
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}
