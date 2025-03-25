#pragma once

#include "rl/info.hpp"
#include "rl/types.hpp"
#include "types.hpp"

#include <itkMattesMutualInformationImageToImageMetricv4.h>
#include <itkMeanSquaresImageToImageMetricv4.h>
#include <itkConjugateGradientLineSearchOptimizerv4.h>
#include <itkCommand.h>
#include <itkRegistrationParameterScalesFromPhysicalShift.h>

namespace merlin {

struct MERLIN {
    MERLIN(ImageType::Pointer fixed, ImageType::RegionType mask);

    auto registerMoving(ImageType::Pointer moving) -> TransformType::Pointer; // Register is a keyword, dummy

private:
    using MetricType = itk::MeanSquaresImageToImageMetricv4<ImageType, ImageType, ImageType, double>;
    // using MetricType = itk::MattesMutualInformationImageToImageMetricv4<ImageType, ImageType, ImageType, double>;
    using LocalOptimizerType = itk::ConjugateGradientLineSearchOptimizerv4;
    using ScalesType = itk::RegistrationParameterScalesFromPhysicalShift<MetricType>;
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
      void SetOptimizer(LocalOptimizerType *optimizer);
      void Execute(itk::Object *caller, const itk::EventObject &event) override;
      void Execute(const itk::Object *, const itk::EventObject &event) override;
    };
    ImageType::Pointer fixed;
    MetricType::Pointer metric;
    LocalOptimizerType::Pointer localOptimizer;
    LocalObserver::Pointer localObserver;
    ScalesType::Pointer scales;

    auto initTransform(ImageType::Pointer moving) -> TransformType::Pointer;
    void initMetric(ImageType::RegionType mask);
    void initOptimizer();
};
} // namespace merlin
