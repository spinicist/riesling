#include "../src/decomp.h"
#include <catch2/catch.hpp>
#include <fmt/ostream.h>

TEST_CASE("Decomp", "[Decomp]")
{
  Log log;

  long const nvar = 8;
  long const nsamp = 64;

  SECTION("PCA")
  {
    Cx2 data(nvar, nsamp);
    data.setRandom();
    Cx2 const gram = Covariance(data);

    CHECK(gram.dimension(0) == nvar);
    CHECK(gram.dimension(1) == nvar);
    CHECK(gram(0, 0).real() > 0.f);
    CHECK(gram(1, 1).real() > 0.f);
    CHECK(gram(0, 0).imag() == Approx(0.f).margin(1.e-3f));
    CHECK(gram(1, 1).imag() == Approx(0.f).margin(1.e-3f));
    CHECK(gram(1, 0).real() == Approx(gram(0, 1).real()).margin(1.e-3f));
    CHECK(gram(1, 0).imag() == Approx(-gram(0, 1).imag()).margin(1.e-3f));

    Cx2 vecs(gram.dimensions());
    R1 vals(gram.dimension(0));
    PCA(gram, vecs, vals, log);
    CHECK(vecs.dimension(0) == gram.dimension(0));
    CHECK(vecs.dimension(1) == gram.dimension(1));
    CHECK(vals.dimension(0) == gram.dimension(0));
  }
}