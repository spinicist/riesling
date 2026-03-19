#include "stack.hpp"

#include "../log/log.hpp"

#include <ranges>

namespace rl::Proxs {

Stack::Stack(std::vector<Ptr> const ps)
  : Prox(fmt::format("{}", ps | std::views::transform([](Ptr const p) { return p->name; })),
         std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, Prox::Ptr const &p) { return i + p->sz; }))
  , proxs{ps}
{
}

Stack::Stack(Prox::Ptr p1, std::vector<Ptr> const ps)
  : Prox(fmt::format("{}+{}", p1->name, fmt::join(ps | std::views::transform([](Ptr const p) { return p->name; }), "+")),
         p1->sz + std::accumulate(ps.begin(), ps.end(), 0L, [](Index const i, Prox::Ptr const &p) { return i + p->sz; }))
  , proxs{p1}
{
  proxs.insert(proxs.end(), ps.begin(), ps.end());
}

auto Stack::Make(std::vector<Ptr> p) -> Ptr { return std::make_shared<Stack>(p); }

void Stack::apply(float const α, Map x) const
{
  Index st = 0;
  for (auto &p : proxs) {
    Map xm(x.data() + st, p->sz);
    p->apply(α, xm);
    st += p->sz;
  }
}

void Stack::conj(float const α, Map x) const
{
  Index st = 0;
  for (auto &p : proxs) {
    Map xm(x.data() + st, p->sz);
    p->conj(α, xm);
    st += p->sz;
  }
}

} // namespace rl::Proxs
