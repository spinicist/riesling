#include "stack.hpp"

#include "../log/log.hpp"

namespace rl::Proxs {

template <typename S> Stack<S>::Stack(std::vector<Ptr> const ps)
  : Prox<S>(std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, Prox<S>::Ptr const &p) { return i + p->sz; }))
  , proxs{ps}
{
}

template <typename S> Stack<S>::Stack(Prox<S>::Ptr p1, std::vector<Ptr> const ps)
  : Prox<S>(p1->sz + std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, Prox<S>::Ptr const &p) { return i + p->sz; }))
  , proxs{p1}
{
  proxs.insert(proxs.end(), ps.begin(), ps.end());
}

template <typename S> auto Stack<S>::Make(std::vector<Ptr> p) -> Ptr { return std::make_shared<Stack<S>>(p); }

template <typename S> void Stack<S>::primal(float const α, CMap x, Map z) const
{
  Index st = 0;
  for (auto &p : proxs) {
    CMap xm(x.data() + st, p->sz);
    Map  zm(z.data() + st, p->sz);
    p->primal(α, xm, zm);
    st += p->sz;
  }
}

template <typename S> void Stack<S>::dual(float const α, CMap x, Map z) const
{
  Index st = 0;
  for (auto &p : proxs) {
    CMap xm(x.data() + st, p->sz);
    Map  zm(z.data() + st, p->sz);
    p->dual(α, xm, zm);
    st += p->sz;
  }
}

template struct Stack<float>;
template struct Stack<Cx>;

} // namespace rl::Proxs
