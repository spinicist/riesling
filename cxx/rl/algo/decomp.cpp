#include "decomp.hpp"

#include "../log/log.hpp"
#include "../tensors.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>

namespace rl {

template <typename S> Eig<S>::Eig(Eigen::Ref<Matrix const> const &g)
{
  if (g.rows() != g.cols()) { throw Log::Failure("Eig", "This is for self-adjoint Eigensystems"); }
  Eigen::SelfAdjointEigenSolver<Matrix> eig(g);
  V = eig.eigenvalues().reverse();
  P = eig.eigenvectors().rowwise().reverse();
}
template struct Eig<float>;
template struct Eig<Cx>;

template <typename S> SVD<S>::SVD(Eigen::Ref<Matrix const> const &mat)
{
  auto const svd = mat.template bdcSvd<Eigen::ComputeThinU | Eigen::ComputeThinV>();
  S = svd.singularValues();
  U = svd.matrixU();
  V = svd.matrixV();
}

template <typename Scalar> auto SVD<Scalar>::variance(Index const N) const -> RealArray
{
  RealArray const squares = S.square();
  RealArray       cumsum(N);
  std::partial_sum(squares.begin(), squares.begin() + N, cumsum.begin());
  cumsum /= squares.sum();
  return cumsum;
}

template <typename Scalar> auto SVD<Scalar>::basis(Index const N, bool const scale) const -> Matrix
{
  if (scale) {
    Matrix basis = (V.leftCols(N) * (S.head(N).log1p() / std::log1p(S(0))).matrix().asDiagonal()).adjoint();
    return basis;
  } else {
    return V.leftCols(N).adjoint();
  }
}

template <typename Scalar> auto SVD<Scalar>::equalized(Index const N) const -> Matrix
{
  // Amoeba Rotation
  // http://stats.stackexchange.com/a/177555/87414
  Log::Print("SVD", "Equalizing variances for first {} vectors", N);
  RealArray       v = S.head(N).square();
  float const     μ = v.mean();
  Eigen::MatrixXf R = Eigen::MatrixXf::Identity(N, N);
  for (Index ii = 0; ii < N - 1; ii++) {
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
  return (V.leftCols(N) * R.cast<Scalar>()).adjoint();
}

template struct SVD<float>;
template struct SVD<Cx>;
template struct SVD<Cxd>;

} // namespace rl
