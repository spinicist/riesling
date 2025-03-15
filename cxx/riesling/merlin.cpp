#include "merlin.hpp"

#include "rl/log.hpp"

#include <flux.hpp>

// Based on antsAI

#include <itkAffineTransform.h>
#include <itkCastImageFilter.h>
#include <itkCenteredTransformInitializer.h>
#include <itkCommand.h>
#include <itkConjugateGradientLineSearchOptimizerv4.h>
#include <itkImageToImageMetricv4.h>
#include <itkImportImageFilter.h>
#include <itkMattesMutualInformationImageToImageMetricv4.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>

#include "vnl/vnl_cross.h"
#include "vnl/vnl_inverse.h"

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

void InitializeTransform(ImageType::Pointer fixed, ImageType::Pointer moving, TransformType::Pointer t)
{
  // This aligns the geometric center of the images, which should be the same anyway, but keep for completeness
  using TransformInitializerType = itk::CenteredTransformInitializer<TransformType, ImageType, ImageType>;
  auto initializer = TransformInitializerType::New();
  initializer->SetTransform(t);
  initializer->SetFixedImage(fixed);
  initializer->SetMovingImage(moving);
  initializer->GeometryOn();
  initializer->InitializeTransform();
}

void AlignMoments(ImageType::Pointer fixed, ImageType::Pointer moving, TransformType::Pointer t)
{
  // Now align the centers of mass and principal axes using the rotation and translation only
  using ImageMomentsCalculatorType = typename itk::ImageMomentsCalculator<ImageType>;
  auto fixedMC = ImageMomentsCalculatorType::New();
  fixedMC->SetImage(fixed);
  fixedMC->Compute();
  auto fixedCoG = fixedMC->GetCenterOfGravity();
  auto fixedPA = fixedMC->GetPrincipalAxes();

  auto movingMC = ImageMomentsCalculatorType::New();
  movingMC->SetImage(moving);
  movingMC->Compute();
  auto movingCoG = movingMC->GetCenterOfGravity();
  auto movingPA = movingMC->GetPrincipalAxes();

  itk::Vector<double, 3> translation;
  for (unsigned int i = 0; i < 3; i++) {
    translation[i] = movingCoG[i] - fixedCoG[i];
  }
  t->SetTranslation(translation);

  /** Solve Wahba's problem --- http://en.wikipedia.org/wiki/Wahba%27s_problem */
  vnl_matrix<double> B;

  auto fixedPrimaryEigenVector = fixedPA.GetVnlMatrix().get_row(2).as_vector();
  auto fixedSecondaryEigenVector = fixedPA.GetVnlMatrix().get_row(1).as_vector();
  auto movingPrimaryEigenVector = movingPA.GetVnlMatrix().get_row(2).as_vector();
  auto movingSecondaryEigenVector = movingPA.GetVnlMatrix().get_row(1).as_vector();

  B = outer_product(movingPrimaryEigenVector, fixedPrimaryEigenVector) +
      outer_product(movingSecondaryEigenVector, fixedSecondaryEigenVector);

  vnl_svd<double>    wahba(B);
  vnl_matrix<double> A = wahba.V() * wahba.U().transpose();
  A = vnl_inverse(A);
  auto det = vnl_determinant(A);

  if (det < 0.0) {
    vnl_matrix<double> I(A);
    I.set_identity();
    for (unsigned int i = 0; i < 3; i++) {
      if (A(i, i) < 0.0) { I(i, i) = -1.0; }
    }
    A = A * I.transpose();
    det = vnl_determinant(A);
    rl::Log::Debug("MERLIN", "New determinant {}", det);
  }
  t->SetMatrix(TransformType::MatrixType(A));
}

using MetricType = itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType, ImageType, double>;
auto SetupMetric(ImageType::Pointer fixed, ImageType::Pointer moving, ImageType::RegionType region, TransformType::Pointer t)
  -> MetricType::Pointer
{
  auto metric = MetricType::New();
  metric->SetNumberOfHistogramBins(64);
  metric->SetUseMovingImageGradientFilter(true);
  metric->SetUseFixedImageGradientFilter(true);
  metric->SetFixedImage(fixed);
  if (flux::all(region.GetSize(), [](int i) { return i > 0; })) {
    metric->SetVirtualDomain(fixed->GetSpacing(), fixed->GetOrigin(), fixed->GetDirection(), region);
  }
  metric->SetMovingImage(moving);
  metric->SetUseSampledPointSet(false);

  using MetricSamplePointSetType = typename MetricType::FixedSampledPointSetType;
  using SamplePointType = typename MetricSamplePointSetType::PointType;
  Index const     stride = 8;
  Index           index = 0;
  Index           id = 0;
  SamplePointType point;
  auto            samplePointSet = MetricSamplePointSetType::New();
  samplePointSet->Initialize();

  itk::ImageRegionConstIteratorWithIndex<ImageType> It(fixed, fixed->GetRequestedRegion());
  for (It.GoToBegin(); !It.IsAtEnd(); ++It) {
    if (index % stride == 0) {
      fixed->TransformIndexToPhysicalPoint(It.GetIndex(), point);
      samplePointSet->SetPoint(id++, point);
    }
    index++;
  }
  metric->SetMovingTransform(t);
  metric->Initialize();
  return metric;
}

using OptimizerType = itk::ConjugateGradientLineSearchOptimizerv4;
class Observer : public itk::Command
{
private:
  itk::WeakPointer<OptimizerType> opt_;

  int total_ = 0, interval_ = 0;

protected:
  Observer(){};

public:
  using Self = Observer;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  itkOverrideGetNameOfClassMacro(Self);
  itkNewMacro(Self);
  void SetOptimizer(OptimizerType *optimizer)
  {
    opt_ = optimizer;
    opt_->AddObserver(itk::IterationEvent(), this);
    opt_->AddObserver(itk::StartEvent(), this);
    opt_->AddObserver(itk::EndEvent(), this);
  }
  void Execute(itk::Object *caller, const itk::EventObject &event) override { Execute((const itk::Object *)caller, event); }
  void Execute(const itk::Object *, const itk::EventObject &event) override
  {
    if (typeid(event) == typeid(itk::StartEvent)) {
      this->total_ = opt_->GetNumberOfIterations();

      if (interval_ == 0) {
        if (total_ > 10) {
          interval_ = total_ / 10;
        } else {
          interval_ = 1;
        }
      }

      rl::Log::Debug("MERLIN", "Optimizer start");
    } else if (typeid(event) == typeid(itk::IterationEvent)) {
      auto const ii = opt_->GetCurrentIteration();
      if (ii % interval_ == 0) {
        rl::Log::Debug("MERLIN", "{:02d} {:.3f} [{:3f}]", ii, opt_->GetValue(), fmt::join(opt_->GetCurrentPosition(), ","));
      }
    } else if (typeid(event) == typeid(itk::EndEvent)) {
      // if (opt_->GetMetricValuesList().size()) {
      //   rl::Log::Debug("MERLIN", "Optimizer finished. Best metric {:.3f}",
      //                  opt_->GetMetricValuesList()[opt_->GetBestParametersIndex()]);
      // } else {
      //   rl::Log::Debug("MERLIN", "Optimizer contained no metric values");
      // }
    }
  }
};

auto SetupOptimizer(MetricType::Pointer metric, TransformType::Pointer t) -> OptimizerType::Pointer
{
  using RegistrationParameterScalesFromPhysicalShiftType = itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
  auto scalesEstimator = RegistrationParameterScalesFromPhysicalShiftType::New();
  scalesEstimator->SetMetric(metric);
  scalesEstimator->SetTransformForward(true);

  RegistrationParameterScalesFromPhysicalShiftType::ScalesType scale(t->GetNumberOfParameters());
  scalesEstimator->EstimateScales(scale);

  auto localOptimizer = OptimizerType::New();
  localOptimizer->SetLowerLimit(0);
  localOptimizer->SetUpperLimit(2);
  localOptimizer->SetEpsilon(0.1);
  localOptimizer->SetMaximumLineSearchIterations(10);
  localOptimizer->SetLearningRate(0.1);
  localOptimizer->SetMaximumStepSizeInPhysicalUnits(0.1);
  localOptimizer->SetNumberOfIterations(64);
  localOptimizer->SetMinimumConvergenceValue(1e-5);
  localOptimizer->SetConvergenceWindowSize(5);
  localOptimizer->SetDoEstimateLearningRateOnce(true);
  localOptimizer->SetScales(scale);
  localOptimizer->SetMetric(metric);

  return localOptimizer;
}

auto Register(ImageType::Pointer fixed, ImageType::Pointer moving, ImageType::RegionType mask) -> TransformType::Pointer
{
  auto t = TransformType::New();
  InitializeTransform(fixed, moving, t);
  // AlignMoments(fixed, moving, t);
  auto metric = SetupMetric(fixed, moving, mask, t);
  auto optimizer = SetupOptimizer(metric, t);
  try {
    Observer::Pointer o = Observer::New();
    o->SetOptimizer(optimizer);
    optimizer->StartOptimization();
  } catch (const itk::ExceptionObject &err) {
    throw rl::Log::Failure("Reg", "{}", err.what());
  }

  auto tfm = TransformType::New();
  tfm->SetParameters(optimizer->GetCurrentPosition());
  return tfm;
}
} // namespace merlin
