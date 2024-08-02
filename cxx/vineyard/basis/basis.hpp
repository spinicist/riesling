#pragma once

#include "types.hpp"

#include <memory>

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

    auto nB() const -> Index;
    auto nSample() const -> Index;
    auto nTrace() const -> Index;

    void write(std::string const &basisFile) const;
    void concat(Basis const &other);

    template <int ND> auto blend(CxN<ND> const &images, Index const is, Index const it) const -> CxN<ND - 1>;
    template <int ND> void applyR(CxN<ND> &data) const;
};

auto LoadBasis(std::string const &basisFile) -> std::unique_ptr<Basis>;

} // namespace rl
