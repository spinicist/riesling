#pragma once

#include "../types.hpp"

#include <memory>

namespace rl {

struct Basis {
    using Ptr = Basis *;
    using CPtr = Basis const *;
    Cx3 B;
    Cx2 R;
    Re1 t;

    Basis();
    Basis(Cx3 const &B, Re1 const &t);
    Basis(Cx3 const &B, Re1 const &t, Cx2 const &R);
    Basis(Index const nB, Index const nSample, Index const nTrace);

    auto nB() const -> Index;
    auto nSample() const -> Index;
    auto nTrace() const -> Index;

    auto entry(Index const sample, Index const trace) const -> Cx1;
    auto entryConj(Index const sample, Index const trace) const -> Cx1;

    void write(std::string const &basisFile) const;
    void concat(Basis const &other);

    template <int ND> auto blend(CxN<ND> const &images, Index const is, Index const it) const -> CxN<ND - 1>;
    template <int ND> void applyR(CxN<ND> &data) const;
};

auto LoadBasis(std::string const &basisFile) -> std::unique_ptr<Basis>;

} // namespace rl
