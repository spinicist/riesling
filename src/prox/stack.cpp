#include "stack.hpp"

namespace rl::Prox {

template <typename S>
StackProx<S>::StackProx(std::vector<std::shared_ptr<Prox<S>>> const p)
  : Prox<S>(std::accumulate(p.begin(), p.end(), 0L, [](Index const i, std::shared_ptr<Prox<S>> const &p) { return i + p->sz; }))
  , proxs{p}
{
}

template <typename S>
void StackProx<S>::apply(float const α, CMap const &x, Map &z) const
{
  Index st = 0;
  for (auto &p : proxs) {
    CMap const xm(x.data() + st, p->sz);
    Map zm(z.data() + st, p->sz);
    p->apply(α, xm, zm);
    st += p->sz;
  }
}

template <typename S>
void StackProx<S>::apply(std::shared_ptr<Ops::Op<S>> const αs1, CMap const &x, Map &z) const
{
  assert(α->ops.size() == proxs.size());

  if (auto const αs = std::dynamic_pointer_cast<Ops::DStack<S>>(αs1)) {
    Index st = 0;
    for (Index ii = 0; ii < proxs.size(); ii++) {
      auto &p = proxs[ii];
      auto &α = αs->ops[ii];
      CMap const xm(x.data() + st, p->sz);
      Map zm(z.data() + st, p->sz);
      p->apply(α, xm, zm);
      st += p->sz;
    }
  } else {
    Log::Fail("C++ is stupid");
  }
}

template struct StackProx<float>;
template struct StackProx<Cx>;

} // namespace rl::Prox
