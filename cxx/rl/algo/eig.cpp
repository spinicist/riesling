#include "eig.hpp"

#include "common.hpp"
#include "iter.hpp"

namespace rl {

auto PowerMethodForward(std::shared_ptr<Ops::Op> A, std::shared_ptr<Ops::Op> P, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power", "A'PA");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->cols());
  float            val = ParallelNorm(vec);
  vec /= val;
  Iterating::Starting();
  for (auto ii = 0; ii < iterLimit; ii++) {
    if (P) {
      vec = A->adjoint(P->adjoint(A->forward(vec)));
    } else {
      vec = A->adjoint(A->forward(vec));
    }
    val = ParallelNorm(vec);
    vec.device(Threads::CoreDevice()) = vec / val;
    Log::Print("Power", "{} Eigenvalue {}", ii, val);
    if (Iterating::ShouldStop("Power")) { break; }
  }
  Iterating::Finished();
  return {val, vec};
}

auto PowerMethodAdjoint(std::shared_ptr<Ops::Op> A, std::shared_ptr<Ops::Op> P, Index const iterLimit) -> PowerReturn
{
  Log::Print("Power", "PAA'");
  Eigen::VectorXcf vec = Eigen::VectorXcf::Random(A->rows());
  float            val = ParallelNorm(vec);
  vec /= val;
  Iterating::Starting();
  for (auto ii = 0; ii < iterLimit; ii++) {
    if (P) {
      vec = P->adjoint(A->forward(A->adjoint(vec)));
    } else {
      vec = A->forward(A->adjoint(vec));
    }
    val = ParallelNorm(vec);
    vec.device(Threads::CoreDevice()) = vec / val;
    Log::Print("Power", "{} Eigenvalue {}", ii, val);
    if (Iterating::ShouldStop("Power")) { break; }
  }
  Iterating::Finished();
  return {val, vec};
}

} // namespace rl
