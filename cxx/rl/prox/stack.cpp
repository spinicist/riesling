#include "stack.hpp"

#include "../log/log.hpp"

namespace rl::Proxs {

template <typename S>
StackProx<S>::StackProx(std::vector<std::shared_ptr<Prox<S>>> const ps)
  : Prox<S>(
      std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, std::shared_ptr<Prox<S>> const &p) { return i + p->sz; }))
  , proxs{ps}
{
}

template <typename S>
StackProx<S>::StackProx(std::shared_ptr<Prox<S>> p1, std::vector<std::shared_ptr<Prox<S>>> const ps)
  : Prox<S>(
      p1->sz +
      std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, std::shared_ptr<Prox<S>> const &p) { return i + p->sz; }))
  , proxs{p1}
{
  proxs.insert(proxs.end(), ps.begin(), ps.end());
}

template <typename S>
void StackProx<S>::apply(float const α, CMap x, Map z) const
{
  Index st = 0;
  for (auto &p : proxs) {
    CMap xm(x.data() + st, p->sz);
    Map        zm(z.data() + st, p->sz);
    p->apply(α, xm, zm);
    st += p->sz;
  }
}

template <typename S>
void StackProx<S>::apply(std::shared_ptr<Ops::Op<S>> const αs1, CMap x, Map z) const
{
  if (auto const αs = std::dynamic_pointer_cast<Ops::DStack<S>>(αs1)) {
    assert(αs->ops.size() == proxs.size());
    Index st = 0;
    for (size_t ii = 0; ii < proxs.size(); ii++) {
      auto      &p = proxs[ii];
      auto      &α = αs->ops[ii];
      CMap xm(x.data() + st, p->sz);
      Map        zm(z.data() + st, p->sz);
      p->apply(α, xm, zm);
      st += p->sz;
    }
  } else {
    throw Log::Failure("Prox", "C++ is stupid");
  }
}

template struct StackProx<float>;
template struct StackProx<Cx>;

} // namespace rl::Proxs
