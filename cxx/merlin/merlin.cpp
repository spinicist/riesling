#include "merlin.hpp"

#include "../args.hpp"
#include "rl/io/hd5.hpp"
#include "rl/log.hpp"
#include "rl/tensors.hpp"

#include "util.hpp"

#include <flux.hpp>

// Based on antsAI

#include <itkAffineTransform.h>

#include <itkCenteredTransformInitializer.h>

#include <itkMultiStartOptimizerv4.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>

using namespace rl;

namespace merlin {

auto MERLIN::initTransform(ImageType::Pointer moving) -> TransformType::Pointer
{
  // This aligns the geometric center of the images, which should be the same anyway, but keep for completeness
  using TransformInitializerType = itk::CenteredTransformInitializer<TransformType, ImageType, ImageType>;
  auto initializer = TransformInitializerType::New();
  auto t = TransformType::New();
  initializer->SetTransform(t);
  initializer->SetFixedImage(fixed);
  initializer->SetMovingImage(moving);
  initializer->GeometryOn();
  initializer->InitializeTransform();
  return t;
}

void MERLIN::initMetric(ImageType::RegionType maskRegion)
{
  metric = MetricType::New();
  // metric->SetNumberOfHistogramBins(64);
  metric->SetUseMovingImageGradientFilter(true);
  metric->SetUseFixedImageGradientFilter(true);
  metric->SetFixedImage(fixed);
  metric->SetMovingImage(fixed);

  using MetricSamplePointSetType = typename MetricType::FixedSampledPointSetType;
  using SamplePointType = typename MetricSamplePointSetType::PointType;
  auto samplePointSet = MetricSamplePointSetType::New();
  samplePointSet->Initialize();
  ImageType::RegionType sampleRegion;
  if (flux::all(maskRegion.GetSize(), [](int i) { return i > 0; })) {
    sampleRegion = maskRegion;
  } else {
    sampleRegion = fixed->GetRequestedRegion();
  }

  itk::ImageRegionConstIteratorWithIndex<ImageType> It(fixed, sampleRegion);
  Index                                             id = 0;
  SamplePointType                                   point;
  for (It.GoToBegin(); !It.IsAtEnd(); ++It) {
    fixed->TransformIndexToPhysicalPoint(It.GetIndex(), point);
    samplePointSet->SetPoint(id++, point);
  }
  metric->SetFixedSampledPointSet(samplePointSet);
  metric->SetUseSampledPointSet(true);
  metric->SetMovingTransform(initTransform(fixed));
  metric->Initialize();
}

void MERLIN::LocalObserver::SetOptimizer(LocalOptimizerType *optimizer)
{
  opt_ = optimizer;
  opt_->AddObserver(itk::IterationEvent(), this);
  opt_->AddObserver(itk::StartEvent(), this);
  opt_->AddObserver(itk::EndEvent(), this);
}

void MERLIN::LocalObserver::Execute(itk::Object *caller, const itk::EventObject &event)
{
  Execute((const itk::Object *)caller, event);
}
void MERLIN::LocalObserver::Execute(const itk::Object *, const itk::EventObject &event)
{
  if (typeid(event) == typeid(itk::StartEvent)) {
    this->total_ = opt_->GetNumberOfIterations();
    interval_ = 1;
    rl::Log::Debug("LOCAL", "Start {:5.3e} [{:5.3e}] LR {}", opt_->GetMetric()->GetValue(),
                   fmt::join(opt_->GetCurrentPosition(), ","), opt_->GetLearningRate());
  } else if (typeid(event) == typeid(itk::IterationEvent)) {
    auto const ii = opt_->GetCurrentIteration();
    if (ii % interval_ == 0) {
      rl::Log::Debug("LOCAL", "{:02d} {:5.3e} [{:5.3e}] LR {}", ii, opt_->GetValue(),
                     fmt::join(opt_->GetCurrentPosition(), ","), opt_->GetLearningRate());
    }
  } else if (typeid(event) == typeid(itk::EndEvent)) {
    rl::Log::Debug("LOCAL", "End {} {:5.3e} [{:5.3e}]", opt_->GetCurrentIteration(), opt_->GetMetric()->GetValue(),
                   fmt::join(opt_->GetCurrentPosition(), ","));
  }
}

void MERLIN::initOptimizer()
{
  scales = ScalesType::New();
  scales->SetMetric(metric);
  scales->SetTransformForward(true);

  localOptimizer = LocalOptimizerType::New();
  localOptimizer->SetLowerLimit(0);
  localOptimizer->SetUpperLimit(2);
  localOptimizer->SetEpsilon(0.1);
  localOptimizer->SetMaximumLineSearchIterations(32);
  localOptimizer->SetLearningRate(1);
  localOptimizer->SetMaximumStepSizeInPhysicalUnits(1);
  localOptimizer->SetNumberOfIterations(128);
  localOptimizer->SetMinimumConvergenceValue(1e-6);
  localOptimizer->SetConvergenceWindowSize(5);
  localOptimizer->SetScalesEstimator(scales);
  localOptimizer->SetDoEstimateLearningRateOnce(true);
  localOptimizer->SetMetric(metric);
}

MERLIN::MERLIN(ImageType::Pointer f, ImageType::RegionType mask)
  : fixed(f)
{
  initMetric(mask);
  initOptimizer();
  localObserver = LocalObserver::New();
  localObserver->SetOptimizer(localOptimizer);
}

auto MERLIN::registerMoving(ImageType::Pointer moving) -> TransformType::Pointer
{
  metric->SetMovingImage(moving);
  metric->SetMovingTransform(initTransform(moving));
  metric->Initialize();
  localOptimizer->SetMetric(metric);
  // localOptimizer->SetLearningRate(0.1);
  try {
    localOptimizer->StartOptimization();
  } catch (const itk::ExceptionObject &err) {
    rl::Log::Warn("MERLIN", "{}", err.what());
  }
  auto t = TransformType::New();
  t->SetParameters(localOptimizer->GetCurrentPosition());
  return t;
}
} // namespace merlin
