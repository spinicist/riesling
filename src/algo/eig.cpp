#include "eig.hpp"

namespace rl {

auto PowerMethodForward(std::shared_ptr<LinOps::Op<Cx>> A, std::shared_ptr<LinOps::Op<Cx>> P, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power Method for A'A");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->cols());
  float val = vec.norm();
  vec /= val;
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = A->adjoint(P->adjoint(A->forward(vec)));
    val = vec.norm();
    vec /= val;
    Log::Print<Log::Level::High>("Iteration {} Eigenvalue {}", ii, val);
  }

  return {val, vec};
}

auto PowerMethodAdjoint(std::shared_ptr<LinOps::Op<Cx>> A, std::shared_ptr<LinOps::Op<Cx>> P, Index const iterLimit) -> PowerReturn
{
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->rows());
  float val = vec.norm();
  vec /= val;
  Log::Print("Power Method for adjoint system AA'");
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = P->adjoint(A->forward(A->adjoint(vec)));
    val = vec.norm();
    vec /= val;
    Log::Print<Log::Level::High>("Iteration {} Eigenvalue {}", ii, val);
  }

  return {val, vec};
}

} // namespace rl
