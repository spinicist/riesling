#define CATCH_CONFIG_ENABLE_BENCHMARKING

#include <catch2/catch.hpp>
#include <unsupported/Eigen/CXX11/Tensor>

TEST_CASE("RSS")
{
  int const sz = 128;
  int const hsz = sz / 2;
  int const qsz = sz / 4;
  using Cx4 = Eigen::Tensor<std::complex<float>, 4>;
  using Cx3 = Eigen::Tensor<std::complex<float>, 3>;
  using R3 = Eigen::Tensor<float, 3>;
  using Dims4 = Cx4::Dimensions;
  Cx4 grid(16, sz, sz, sz);
  grid.setRandom();
  Cx3 xsum(hsz, hsz, hsz);
  xsum.setZero();
  R3 sum(hsz, hsz, hsz);
  sum.setZero();
  Eigen::ThreadPool gp(std::thread::hardware_concurrency());
  Eigen::ThreadPoolDevice dev(&gp, gp.NumThreads());

  BENCHMARK("Naive")
  {
    xsum.device(dev) =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .sum(Eigen::array<int, 1>{0})
            .sqrt();
  };

  BENCHMARK("Naive-Real")
  {
    sum.device(dev) =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .real()
            .sum(Eigen::array<int, 1>{0})
            .sqrt();
  };

  BENCHMARK("IndexList1")
  {
    xsum.device(dev) =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .sum(Eigen::IndexList<Eigen::type2index<0>>())
            .sqrt();
  };

  BENCHMARK("IndexList1-Real")
  {
    sum.device(dev) =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .real()
            .sum(Eigen::IndexList<Eigen::type2index<0>>())
            .sqrt();
  };

  BENCHMARK("IndexList1-Real2")
  {
    sum.device(dev) =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .sum(Eigen::IndexList<Eigen::type2index<0>>())
            .real()
            .sqrt();
  };

  BENCHMARK("IndexList1-Real3")
  {
    sum.device(dev) =
        (grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
         grid.slice(Dims4{0, qsz, qsz, qsz}, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
            .sum(Eigen::IndexList<Eigen::type2index<0>>())
            .sqrt()
            .real();
  };

  BENCHMARK("IndexList2")
  {
    Eigen::IndexList<Eigen::type2index<0>, int, int, int> start;
    start.set(1, qsz);
    start.set(2, qsz);
    start.set(3, qsz);
    xsum.device(dev) = (grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
                        grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
                           .sum(Eigen::IndexList<Eigen::type2index<0>>())
                           .sqrt();
  };

  BENCHMARK("IndexList2-Real")
  {
    Eigen::IndexList<Eigen::type2index<0>, int, int, int> start;
    start.set(1, qsz);
    start.set(2, qsz);
    start.set(3, qsz);
    sum.device(dev) = (grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz}) *
                       grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz}).conjugate())
                          .real()
                          .sum(Eigen::IndexList<Eigen::type2index<0>>())
                          .sqrt();
  };

  BENCHMARK("UnaryOp")
  {
    Eigen::IndexList<Eigen::type2index<0>, int, int, int> start;
    start.set(1, qsz);
    start.set(2, qsz);
    start.set(3, qsz);
    sum.device(dev) = grid.slice(start, Dims4{grid.dimension(0), hsz, hsz, hsz})
                          .unaryExpr(Eigen::internal::scalar_abs2_op<std::complex<float>>())
                          .sum(Eigen::IndexList<Eigen::type2index<0>>())
                          .sqrt();
  };
}