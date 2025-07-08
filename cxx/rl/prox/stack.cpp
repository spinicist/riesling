#include "stack.hpp"

#include "../log/log.hpp"

namespace rl::Proxs {

Stack::Stack(std::vector<Ptr> const ps)
  : Prox(std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, Prox::Ptr const &p) { return i + p->sz; }))
  , proxs{ps}
{
}

Stack::Stack(Prox::Ptr p1, std::vector<Ptr> const ps)
  : Prox(p1->sz + std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, Prox::Ptr const &p) { return i + p->sz; }))
  , proxs{p1}
{
  proxs.insert(proxs.end(), ps.begin(), ps.end());
}

auto Stack::Make(std::vector<Ptr> p) -> Ptr { return std::make_shared<Stack>(p); }

void Stack::apply(float const α, CMap x, Map z) const
{
  Index st = 0;
  for (auto &p : proxs) {
    CMap xm(x.data() + st, p->sz);
    Map  zm(z.data() + st, p->sz);
    p->apply(α, xm, zm);
    st += p->sz;
  }
}

void Stack::conj(float const α, CMap x, Map z) const
{
  Index st = 0;
  for (auto &p : proxs) {
    CMap xm(x.data() + st, p->sz);
    Map  zm(z.data() + st, p->sz);
    p->conj(α, xm, zm);
    st += p->sz;
  }
}

} // namespace rl::Proxs
