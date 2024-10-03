#include "eig.hpp"

#include "algo/common.hpp"

namespace rl {

auto PowerMethod(std::shared_ptr<Ops::Op<Cx>> A, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power", "A'A");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->cols());
  float            val = ParallelNorm(vec);
  vec /= val;
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = A->adjoint(A->forward(vec));
    val = ParallelNorm(vec);
    vec /= val;
    Log::Print("Power", "{} Eigenvalue {}", ii, val);
  }

  return {val, vec};
}

auto PowerMethodForward(std::shared_ptr<Ops::Op<Cx>> A, std::shared_ptr<Ops::Op<Cx>> P, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power", "A'PA");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->cols());
  float            val = ParallelNorm(vec);
  vec /= val;
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = A->adjoint(P->adjoint(A->forward(vec)));
    val = ParallelNorm(vec);
    vec /= val;
    Log::Print("Power", "{} Eigenvalue {}", ii, val);
  }

  return {val, vec};
}

auto PowerMethodAdjoint(std::shared_ptr<Ops::Op<Cx>> A, std::shared_ptr<Ops::Op<Cx>> P, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power", "PAA'");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->rows());
  float            val = ParallelNorm(vec);
  vec /= val;
  for (auto ii = 0; ii < iterLimit; ii++) {
    vec = P->adjoint(A->forward(A->adjoint(vec)));
    val = ParallelNorm(vec);
    vec /= val;
    Log::Print("Power", "{} Eigenvalue {}", ii, val);
  }

  return {val, vec};
}

} // namespace rl
