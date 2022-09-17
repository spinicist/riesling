#include "dict.hpp"

#include "tensorOps.hpp"
#include "threads.hpp"

#include "log.hpp"

namespace rl {

DictionaryProjection::DictionaryProjection(Re2 d)
  : Functor<Cx4>()
  , dictionary{d}
{
}

auto DictionaryProjection::operator()(Cx4 const &x) const -> Cx4
{
  Cx4 y(x.dimensions());
  fmt::print("x {} dict {}\n", x.dimensions(), dictionary.dimensions());
  auto ztask = [&](Index const iz) {
    for (Index iy = 0; iy < x.dimension(2); iy++) {
      for (Index ix = 0; ix < x.dimension(1); ix++) {
        Cx1 const p = x.chip<3>(iz).chip<2>(iy).chip<1>(ix);
        Index index = 0;
        Cx bestCorr{0.f, 0.f};
        float bestAbsCorr = 0;

        for (Index in = 0; in < dictionary.dimension(0); in++) {
          Re1 const atom = dictionary.chip<0>(in);
          Cx const corr = Dot(atom.cast<Cx>(), p);
          if (std::abs(corr) > bestAbsCorr) {
            bestAbsCorr = std::abs(corr);
            bestCorr = corr;
            index = in;
          }
        }
        y.chip<3>(iz).chip<2>(iy).chip<1>(ix) = bestCorr * dictionary.chip<0>(index).cast<Cx>();
        // y.chip<3>(iz).chip<2>(iy).chip<1>(ix) = p / p.constant(std::polar(Norm(p), std::arg(p(0))));
      }
    }
  };
  Threads::For(ztask, x.dimension(3), "Dictionary Projection");
  return y;
}

} // namespace rl