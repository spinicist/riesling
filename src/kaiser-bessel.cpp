#include "fmt/ostream.h"
#include "kaiser-bessel.h"

#include <cmath>
#include <vector>

/* This code is modified from Boost bessel_i0.hpp, Copyright (c) 2006 Xiaogang Zhang
 * Use, modification and distribution are subject to the
 *  Boost Software License, Version 1.0. (See accompanying file
 *  LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
 */

template <class T, size_t N>
inline T evaluate_polynomial(std::array<T, N> const &poly, T const &z)
{
  T sum = poly[N - 1];
  for (int i = N - 2; i >= 0; --i) {
    sum *= z;
    sum += poly[i];
  }
  return sum;
}

template <typename T>
T bessel_i0(T x)
{
  static const std::array<T, 15> P1 = {
      static_cast<T>(-2.2335582639474375249e+15L),
      static_cast<T>(-5.5050369673018427753e+14L),
      static_cast<T>(-3.2940087627407749166e+13L),
      static_cast<T>(-8.4925101247114157499e+11L),
      static_cast<T>(-1.1912746104985237192e+10L),
      static_cast<T>(-1.0313066708737980747e+08L),
      static_cast<T>(-5.9545626019847898221e+05L),
      static_cast<T>(-2.4125195876041896775e+03L),
      static_cast<T>(-7.0935347449210549190e+00L),
      static_cast<T>(-1.5453977791786851041e-02L),
      static_cast<T>(-2.5172644670688975051e-05L),
      static_cast<T>(-3.0517226450451067446e-08L),
      static_cast<T>(-2.6843448573468483278e-11L),
      static_cast<T>(-1.5982226675653184646e-14L),
      static_cast<T>(-5.2487866627945699800e-18L),
  };
  static const std::array<T, 6> Q1 = {
      static_cast<T>(-2.2335582639474375245e+15L),
      static_cast<T>(7.8858692566751002988e+12L),
      static_cast<T>(-1.2207067397808979846e+10L),
      static_cast<T>(1.0377081058062166144e+07L),
      static_cast<T>(-4.8527560179962773045e+03L),
      static_cast<T>(1.0L),
  };
  static const std::array<T, 7> P2 = {
      static_cast<T>(-2.2210262233306573296e-04L),
      static_cast<T>(1.3067392038106924055e-02L),
      static_cast<T>(-4.4700805721174453923e-01L),
      static_cast<T>(5.5674518371240761397e+00L),
      static_cast<T>(-2.3517945679239481621e+01L),
      static_cast<T>(3.1611322818701131207e+01L),
      static_cast<T>(-9.6090021968656180000e+00L),
  };
  static const std::array<T, 8> Q2 = {
      static_cast<T>(-5.5194330231005480228e-04L),
      static_cast<T>(3.2547697594819615062e-02L),
      static_cast<T>(-1.1151759188741312645e+00L),
      static_cast<T>(1.3982595353892851542e+01L),
      static_cast<T>(-6.0228002066743340583e+01L),
      static_cast<T>(8.5539563258012929600e+01L),
      static_cast<T>(-3.1446690275135491500e+01L),
      static_cast<T>(1.0L),
  };
  T value, factor, r;

  if (x < 0) {
    x = -x; // even function
  }
  if (x == 0) {
    return static_cast<T>(1);
  }
  if (x <= 15) // x in (0, 15]
  {
    T y = x * x;
    value = evaluate_polynomial(P1, y) / evaluate_polynomial(Q1, y);
  } else // x in (15, \infty)
  {
    T y = 1 / x - T(1) / 15;
    r = evaluate_polynomial(P2, y) / evaluate_polynomial(Q2, y);
    factor = exp(x) / sqrt(x);
    value = factor * r;
  }

  return value;
}

Eigen::ArrayXf KB(Eigen::ArrayXf const &x, float const &beta)
{
  Eigen::ArrayXf u = (1. - (x.abs() * 2.f).pow(2.f)).sqrt() * beta;
  Eigen::ArrayXf val = u.unaryExpr([](float const ui) { return bessel_i0(ui); });
  // fmt::print(FMT_STRING("KB x {} u {} val {}\n"), x.transpose(), u.transpose(), val.transpose());
  return val;
}

inline float KB(float const x, float const beta)
{
  if (std::fabsf(x) > 0.5f) {
    return 0.f;
  }
  float const u = sqrtf(1. - 4.f * x * x) * beta;
  return bessel_i0(u);
}

float KB_FT(float const &x, float const &beta)
{
  double const a = sqrt(pow(beta, 2.) - pow(M_PI * x, 2.));
  return a / sinh(a); // * bessel_i0(beta);
}

R2 KBKernel(Point2 const &offset, long const w, float const beta)
{
  long const hw = w / 2;

  R1 kX(w), kY(w);
  for (long ii = 0; ii < w; ii++) {
    kX(ii) = KB(ii - hw - offset(0), beta);
    kY(ii) = KB(ii - hw - offset(1), beta);
  }

  R2 k = kX.reshape(Sz2{w, 1}).broadcast(Sz2{1, w}) * kY.reshape(Sz2{1, w}).broadcast(Sz2{w, 1});
  k /= k.sum();
  return k;
}

R3 KBKernel(Point3 const &offset, long const w, float const beta)
{
  long const hw = w / 2;

  R1 kX(w), kY(w), kZ(w);
  for (long ii = 0; ii < w; ii++) {
    kX(ii) = KB((ii - hw - offset(0)) / w, beta);
    kY(ii) = KB((ii - hw - offset(1)) / w, beta);
    kZ(ii) = KB((ii - hw - offset(2)) / w, beta);
  }

  R3 k = kX.reshape(Sz3{w, 1, 1}).broadcast(Sz3{1, w, w}) *
         kY.reshape(Sz3{1, w, 1}).broadcast(Sz3{w, 1, w}) *
         kZ.reshape(Sz3{1, 1, w}).broadcast(Sz3{w, w, 1});
  k = k / sum(k);
  return k;
}
