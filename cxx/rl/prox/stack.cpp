#include "stack.hpp"

#include "../log/log.hpp"

namespace rl::Proxs {

template <typename S> StackProx<S>::StackProx(std::vector<std::shared_ptr<Prox<S>>> const ps)
  : Prox<S>(
      std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, std::shared_ptr<Prox<S>> const &p) { return i + p->sz; }))
  , proxs{ps}
{
}

template <typename S> StackProx<S>::StackProx(std::shared_ptr<Prox<S>> p1, std::vector<std::shared_ptr<Prox<S>>> const ps)
  : Prox<S>(p1->sz + std::accumulate(
                       ps.begin(), ps.end(), 0L, [](Index const i, std::shared_ptr<Prox<S>> const &p) { return i + p->sz; }))
  , proxs{p1}
{
  proxs.insert(proxs.end(), ps.begin(), ps.end());
}

template <typename S> void StackProx<S>::primal(float const α, CMap x, Map z) const
{
  Index st = 0;
  for (auto &p : proxs) {
    CMap xm(x.data() + st, p->sz);
    Map  zm(z.data() + st, p->sz);
    p->primal(α, xm, zm);
    st += p->sz;
  }
}

template <typename S> void StackProx<S>::dual(float const α, CMap x, Map z) const
{
  Index st = 0;
  for (auto &p : proxs) {
    CMap xm(x.data() + st, p->sz);
    Map  zm(z.data() + st, p->sz);
    p->dual(α, xm, zm);
    st += p->sz;
  }
}

template struct StackProx<float>;
template struct StackProx<Cx>;

} // namespace rl::Proxs
