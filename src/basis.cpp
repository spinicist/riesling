#include "basis.hpp"

#include "algo/decomp.hpp"
#include "algo/stats.hpp"

namespace rl {

auto IdBasis() -> Re2
{
  Re2 id(1, 1);
  id.setConstant(1.f);
  return id;
}

void SaveBasis(
  Eigen::ArrayXXf const &dynamics,
  float const            thresh,
  Index const            nBasis,
  bool const             demean,
  bool const             rotate,
  bool const             normalize,
  HD5::Writer           &writer)
{
  // Calculate SVD - observations are in cols
  Eigen::ArrayXXf d = normalize ? dynamics.colwise().normalized() : dynamics;
  if (demean) { d = d.colwise() - d.rowwise().mean(); }
  auto const svd = SVD<float>(d.transpose());
  Index nRetain = 0;
  if (nBasis) {
    nRetain = nBasis;
  } else {
    nRetain = Threshold(svd.S.square(), thresh);
  }
  Log::Print("Retaining {} basis vectors", nRetain);

  Eigen::MatrixXf basis = svd.V.leftCols(nRetain);
  if (rotate) {
    // Amoeba Rotation
    // http://stats.stackexchange.com/a/177555/87414
    Eigen::ArrayXf  v = svd.S.head(nRetain).square();
    float const     μ = v.mean();
    Eigen::MatrixXf R = Eigen::MatrixXf::Identity(nRetain, nRetain);
    for (Index ii = 0; ii < nRetain - 1; ii++) {
      Index           maxInd = 0, minInd = 0;
      auto const      maxVal = v.maxCoeff(&maxInd);
      auto const      minVal = v.minCoeff(&minInd);
      float const     w = (μ - minVal) / (maxVal - minVal);
      float const     cosθ = std::sqrt(w);
      float const     sinθ = std::sqrt(1.f - w);
      Eigen::Matrix2f rot;
      rot << cosθ, sinθ, //
        -sinθ, cosθ;

      R(Eigen::placeholders::all, std::vector<Index>{maxInd, minInd}) =
        R(Eigen::placeholders::all, std::vector<Index>{maxInd, minInd}) * rot;
      v(maxInd) = μ;
      v(minInd) = maxVal + minVal - μ;
    }
    Eigen::ArrayXf cumv(v.rows());
    std::partial_sum(v.begin(), v.end(), cumv.begin());
    cumv = 100.f * cumv / cumv.tail(1)[0];
    Log::Print("Amoeba Rotation. New variances {}", cumv.transpose());
    basis = basis * R;
  }

  Log::Print("Computing dictionary");
  basis *= std::sqrt(basis.rows());
  Eigen::MatrixXf dict = basis.transpose() * dynamics.matrix();
  Eigen::ArrayXf  norm = dict.colwise().norm();
  dict = dict.array().rowwise() / norm.transpose();

  writer.writeMatrix(Eigen::MatrixXf(basis.transpose()), HD5::Keys::Basis);
  writer.writeMatrix(dict, HD5::Keys::Dictionary);
  writer.writeMatrix(norm, HD5::Keys::Norm);
  writer.writeMatrix(dynamics, HD5::Keys::Dynamics);
}

} // namespace rl
