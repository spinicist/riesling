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
#include <itkMultiStartOptimizerv4.h>
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
auto SetupMetric(ImageType::Pointer     fixed,
                 ImageType::Pointer     moving,
                 ImageType::RegionType  maskRegion,
                 TransformType::Pointer t) -> MetricType::Pointer
{
  auto metric = MetricType::New();
  metric->SetNumberOfHistogramBins(64);
  metric->SetUseMovingImageGradientFilter(true);
  metric->SetUseFixedImageGradientFilter(true);
  metric->SetFixedImage(fixed);
  metric->SetMovingImage(moving);

  using MetricSamplePointSetType = typename MetricType::FixedSampledPointSetType;
  using SamplePointType = typename MetricSamplePointSetType::PointType;
  Index const     stride = 1;
  Index           index = 0;
  Index           id = 0;
  SamplePointType point;
  auto            samplePointSet = MetricSamplePointSetType::New();
  samplePointSet->Initialize();
  ImageType::RegionType sampleRegion;
  if (flux::all(maskRegion.GetSize(), [](int i) { return i > 0; })) {
    sampleRegion = maskRegion;
  } else {
    sampleRegion = fixed->GetRequestedRegion();
  }

  itk::ImageRegionConstIteratorWithIndex<ImageType> It(fixed, sampleRegion);
  for (It.GoToBegin(); !It.IsAtEnd(); ++It) {
    if (index % stride == 0) {
      fixed->TransformIndexToPhysicalPoint(It.GetIndex(), point);
      samplePointSet->SetPoint(id++, point);
    }
    index++;
  }
  metric->SetFixedSampledPointSet(samplePointSet);
  metric->SetUseSampledPointSet(true);
  metric->SetMovingTransform(t);
  metric->Initialize();
  return metric;
}

using LocalOptimizerType = itk::ConjugateGradientLineSearchOptimizerv4;
class LocalObserver : public itk::Command
{
private:
  itk::WeakPointer<LocalOptimizerType> opt_;

  int total_ = 0, interval_ = 0;

protected:
  LocalObserver(){};

public:
  using Self = LocalObserver;
  using Superclass = itk::Command;
  using Pointer = itk::SmartPointer<Self>;
  using ConstPointer = itk::SmartPointer<const Self>;

  itkOverrideGetNameOfClassMacro(Self);
  itkNewMacro(Self);
  void SetOptimizer(LocalOptimizerType *optimizer)
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

      // if (interval_ == 0) {
      //   if (total_ > 10) {
      //     interval_ = total_ / 10;
      //   } else {
      //     interval_ = 1;
      //   }
      // }
      interval_ = 1;

      rl::Log::Debug("LOCAL", "Local optimizer start [{:.3f}]", fmt::join(opt_->GetCurrentPosition(), ","));
    } else if (typeid(event) == typeid(itk::IterationEvent)) {
      auto const ii = opt_->GetCurrentIteration();
      if (ii % interval_ == 0) {
        rl::Log::Debug("LOCAL", "{:02d} {:.3f} [{:3f}]", ii, opt_->GetValue(), fmt::join(opt_->GetCurrentPosition(), ","));
      }
    } else if (typeid(event) == typeid(itk::EndEvent)) {
      rl::Log::Debug("LOCAL", "Local Optimizer finished [{:.3f}]", fmt::join(opt_->GetCurrentPosition(), ","));
    }
  }
};

using OptimizerType = itk::MultiStartOptimizerv4;
class MultiStartObserver : public itk::Command
{
private:
  itk::WeakPointer<OptimizerType> opt_;

  int total_ = 0, interval_ = 0;

protected:
  MultiStartObserver(){};

public:
  using Self = MultiStartObserver;
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

      // if (interval_ == 0) {
      //   if (total_ > 10) {
      //     interval_ = total_ / 10;
      //   } else {
      //     interval_ = 1;
      //   }
      // }
      interval_ = 1;

      rl::Log::Debug("MERLIN", "Optimizer start");
    } else if (typeid(event) == typeid(itk::IterationEvent)) {
      auto const ii = opt_->GetCurrentIteration();
      if (ii % interval_ == 0) {
        rl::Log::Debug("MERLIN", "{:02d} {:.3f} [{:3f}]", ii, opt_->GetValue(), fmt::join(opt_->GetCurrentPosition(), ","));
      }
    } else if (typeid(event) == typeid(itk::EndEvent)) {
      if (opt_->GetMetricValuesList().size()) {
        rl::Log::Debug("MERLIN", "Optimizer finished. Best metric {:.3f} Best Parameters [{:.3f}]",
                       opt_->GetMetricValuesList()[opt_->GetBestParametersIndex()], fmt::join(opt_->GetBestParameters(), ","));
      } else {
        rl::Log::Debug("MERLIN", "Optimizer contained no metric values");
      }
    }
  }
};

struct Both {
  OptimizerType::Pointer mso;
  LocalOptimizerType::Pointer lo;
};
auto SetupOptimizer(MetricType::Pointer metric, TransformType::Pointer t) -> Both
{
  using RegistrationParameterScalesFromPhysicalShiftType = itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
  auto scalesEstimator = RegistrationParameterScalesFromPhysicalShiftType::New();
  scalesEstimator->SetMetric(metric);
  scalesEstimator->SetTransformForward(true);

  RegistrationParameterScalesFromPhysicalShiftType::ScalesType scale(t->GetNumberOfParameters());
  scalesEstimator->EstimateScales(scale);

  auto localOptimizer = LocalOptimizerType::New();
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

  auto multiStartOptimizer = OptimizerType::New();
  multiStartOptimizer->SetScales(scale);
  multiStartOptimizer->SetMetric(metric);
  int          trialCounter = 0;
  auto         parametersList = multiStartOptimizer->GetParametersList();
  double const aMax = 0;
  double const aStep = 1;
  double const tMax = 50;
  double const tStep = 50;
  double const eps = 1e-6;
  using AffineTransformType = itk::AffineTransform<double, 3>;
  auto affine = AffineTransformType::New();

  itk::Vector<double, 3> axis1(0.0);
  itk::Vector<double, 3> axis2(0.0);
  axis1[0] = 1.0;
  axis2[1] = 1.0;
  parametersList.push_back(t->GetParameters());
  parametersList.push_back(t->GetParameters());
  fmt::print(stderr, "t\n{}\n", fmt::streamed(t->GetParameters()));
  for (double a1 = -aMax; a1 <= aMax + eps; a1 += aStep) {
    for (double a2 = -aMax; a2 <= aMax + eps; a2 += aStep) {
      for (double a3 = -aMax; a3 <= aMax + eps; a3 += aStep) {
        for (double t1 = -tMax; t1 <= tMax + eps; t1 += tStep) {
          for (double t2 = -tMax; t2 <= tMax + eps; t2 += tStep) {
            for (double t3 = -tMax; t3 <= tMax + eps; t3 += tStep) {
              AffineTransformType::OutputVectorType st;
              st[0] = t1;
              st[1] = t2;
              st[2] = t3;

              // affine->SetIdentity();
              // affine->SetCenter(t->GetCenter());
              // affine->SetOffset(t->GetOffset());
              // affine->SetMatrix(t->GetMatrix());
              // affine->Translate(st, 0);
              // affine->Rotate3D(axis1, a1, 1);
              // affine->Rotate3D(axis2, a2, 1);
              // affine->Rotate3D(axis1, a3, 1);

              auto search = TransformType::New();
              // search->SetParameters(t->GetParameters());
              search->SetFixedParameters(t->GetFixedParameters());
              // search->SetIdentity();
              // search->SetCenter(t->GetCenter());
              // search->SetOffset(t->GetOffset());
              // search->SetMatrix(affine->GetMatrix());
              search->Translate(st, 0);
              fmt::print(stderr, "search {} {}\n", trialCounter, fmt::streamed(search->GetParameters()));
              parametersList.push_back(search->GetParameters());
              trialCounter++;
            }
          }
        }
      }
    }
  }
  multiStartOptimizer->SetParametersList(parametersList);
  multiStartOptimizer->SetLocalOptimizer(localOptimizer);
  rl::Log::Debug("MERLIN", "Search list has {} entries", trialCounter);
  return {multiStartOptimizer, localOptimizer};
}

auto Register(ImageType::Pointer fixed, ImageType::Pointer moving, ImageType::RegionType mask) -> TransformType::Pointer
{
  auto t = TransformType::New();
  InitializeTransform(fixed, moving, t);
  // AlignMoments(fixed, moving, t);
  auto metric = SetupMetric(fixed, moving, mask, t);
  auto opts = SetupOptimizer(metric, t);
  try {
    auto mso = MultiStartObserver::New();
    auto lo = LocalObserver::New();
    mso->SetOptimizer(opts.mso);
    lo->SetOptimizer(opts.lo);
    opts.mso->StartOptimization();
    exit(EXIT_SUCCESS);
  } catch (const itk::ExceptionObject &err) {
    throw rl::Log::Failure("Reg", "{}", err.what());
  }

  auto tfm = TransformType::New();
  tfm->SetParameters(opts.mso->GetBestParameters());
  return tfm;
}
} // namespace merlin
