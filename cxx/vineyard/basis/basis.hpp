#pragma once

#include "types.hpp"

namespace rl {

struct Basis {
    using Ptr = Basis *;
    using CPtr = Basis const *;
    Cx3 B;
    Cx2 R;

    Basis();
    Basis(Cx3 const &B);
    Basis(Cx3 const &B, Cx2 const &R);
    Basis(Index const nB, Index const nSample, Index const nTrace);
    Basis(std::string const &basisFile);

    void write(std::string const &basisFile) const;
    auto nB() const -> Index;
    auto nSample() const -> Index;
    auto nTrace() const -> Index;

    template <int ND> auto blend(CxN<ND> const &images, Index const is, Index const it) const -> CxN<ND - 1>;
    template <int ND> void applyR(CxN<ND> &data) const;
};

} // namespace rl
