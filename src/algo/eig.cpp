#include "eig.hpp"

namespace rl {

auto PowerMethodForward(std::shared_ptr<LinOps::Op<Cx>> op, std::shared_ptr<LinOps::Op<Cx>> M, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power Method for A'A");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(op->rows());
  float val = vec.norm();
  vec /= val;
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = op->adjoint(M->adjoint(op->forward(vec)));
    val = vec.norm();
    vec /= val;
    Log::Print<Log::Level::High>(FMT_STRING("Iteration {} Eigenvalue {}"), ii, val);
  }

  return {val, vec};
}

auto PowerMethodAdjoint(std::shared_ptr<LinOps::Op<Cx>> op, std::shared_ptr<LinOps::Op<Cx>> M, Index const iterLimit) -> PowerReturn
{
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(op->cols());
  float val = vec.norm();
  vec /= val;
  Log::Print(FMT_STRING("Power Method for adjoint system (AA')"));
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = op->forward(op->adjoint(M->adjoint(vec)));
    val = vec.norm();
    vec /= val;
    Log::Print<Log::Level::High>(FMT_STRING("Iteration {} Eigenvalue {}"), ii, val);
  }

  return {val, vec};
}

} // namespace rl
