#include "hermitian.hpp"

#include "../log/log.hpp"
#include "../sys/threads.hpp"
#include "../tensors.hpp"

namespace rl::Proxs {

Hermitian::Hermitian(float const l, Sz5 const sh)
  : Prox<Cx>(Product(sh))
  , λ{l}
  , shape{sh}
  , F{shape}
{
  λ *= std::sqrt(Product(LastN<4>(shape)));
  Log::Print("Prox", "Hermitian λ {} scaled λ {} Shape {}", l, λ, shape);
}

void Hermitian::apply(float const α, CMap xin, Map zin) const
{
  Eigen::TensorMap<Cx5 const> const x(xin.data(), shape);
  Eigen::TensorMap<Cx5>             z(zin.data(), shape);
  float const                       t = λ * α;

  // auto const Fx = F.forward(x);
  auto sz = x.dimensions();
  sz[2] -= 1;
  sz[3] -= 1;
  sz[4] -= 1;
  Cx5 u(shape);
  u.setZero();
  u.slice(Sz5{0, 0, 1, 1, 1}, sz) =
    (x.slice(Sz5{0, 0, 1, 1, 1}, sz) +
     x.slice(Sz5{0, 0, 1, 1, 1}, sz).reverse(Eigen::array<bool, 5>{false, false, true, true, true}).conjugate()) /
    x.constant(2.f);
  for (Index ii = 0; ii < u.dimension(0); ii++) {
    auto       uc = u.chip<0>(ii);
    auto const normuc = Norm<true>(uc);
    if (normuc > t) {
      uc = uc * uc.constant(1.f - t / normuc);
    } else {
      uc.setZero();
    }
  }
  // z = F.adjoint(u);
  z = u;
  Log::Debug("Prox", "Hermitian α {} λ {} t {} |x| {} |z| {}", α, λ, t, Norm<true>(x), Norm<true>(z));
}

} // namespace rl::Proxs
