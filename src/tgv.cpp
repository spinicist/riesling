#include "tgv.h"

#include "tensorOps.h"
#include "threads.h"

auto fdiff(Cx3 const &a, Eigen::Index const d)
{
  Dims3 const sz{a.dimension(0) - 2, a.dimension(1) - 2, a.dimension(2) - 2};
  Dims3 const st1{1, 1, 1};
  Dims3 fwd{1, 1, 1};
  fwd[d] = 2;

  return (a.slice(fwd, sz) - a.slice(st1, sz));
};

auto bdiff(Cx3 const &a, Eigen::Index const d)
{
  Dims3 const sz{a.dimension(0) - 2, a.dimension(1) - 2, a.dimension(2) - 2};
  Dims3 const st1{1, 1, 1};
  Dims3 bck{1, 1, 1};
  bck[d] = 0;

  return (a.slice(st1, sz) - a.slice(bck, sz));
};

auto cdiff(Cx3 const &a, Eigen::Index const d)
{
  Dims3 const sz{a.dimension(0) - 2, a.dimension(1) - 2, a.dimension(2) - 2};
  Dims3 const st1{1, 1, 1};
  Dims3 fwd{1, 1, 1};
  Dims3 bck{1, 1, 1};
  fwd[d] = 2;
  bck[d] = 0;

  return (a.slice(fwd, sz) - a.slice(bck, sz)) / a.slice(st1, sz).constant(2.f);
};

void calc_scalar_grad(Cx3 const &a, Cx4 &g)
{
  Dims3 const sz{a.dimension(0) - 2, a.dimension(1) - 2, a.dimension(2) - 2};
  Dims3 const st1{1, 1, 1};
  g.chip<3>(0).slice(st1, sz).device(Threads::GlobalDevice()) = fdiff(a, 0);
  g.chip<3>(1).slice(st1, sz).device(Threads::GlobalDevice()) = fdiff(a, 1);
  g.chip<3>(2).slice(st1, sz).device(Threads::GlobalDevice()) = fdiff(a, 2);
};

void calc_vector_grad(Cx4 const &x, Cx4 &gx)
{
  Dims3 const sz{x.dimension(0) - 2, x.dimension(1) - 2, x.dimension(2) - 2};
  Dims3 const st1{1, 1, 1};

  gx.chip<3>(0).slice(st1, sz).device(Threads::GlobalDevice()) = bdiff(x.chip<3>(0), 0);
  gx.chip<3>(1).slice(st1, sz).device(Threads::GlobalDevice()) = bdiff(x.chip<3>(1), 1);
  gx.chip<3>(2).slice(st1, sz).device(Threads::GlobalDevice()) = bdiff(x.chip<3>(2), 2);

  gx.chip<3>(3).slice(st1, sz).device(Threads::GlobalDevice()) =
      (bdiff(x.chip<3>(0), 1) + bdiff(x.chip<3>(1), 0)) /
      gx.chip<3>(3).slice(st1, sz).constant(2.f);

  gx.chip<3>(4).slice(st1, sz).device(Threads::GlobalDevice()) =
      (bdiff(x.chip<3>(0), 2) + bdiff(x.chip<3>(2), 0)) /
      gx.chip<3>(4).slice(st1, sz).constant(2.f);

  gx.chip<3>(5).slice(st1, sz).device(Threads::GlobalDevice()) =
      (bdiff(x.chip<3>(1), 2) + bdiff(x.chip<3>(2), 1)) /
      gx.chip<3>(5).slice(st1, sz).constant(2.f);
}

inline void calc_vector_div(Cx4 const &x, Cx3 &div)
{
  Dims3 const sz{x.dimension(0) - 2, x.dimension(1) - 2, x.dimension(2) - 2};
  Dims3 const st1{1, 1, 1};
  div.slice(st1, sz).device(Threads::GlobalDevice()) =
      bdiff(x.chip<3>(0), 0) + bdiff(x.chip<3>(1), 1) + bdiff(x.chip<3>(2), 2);
}

inline void calc_tensor_div(Cx4 const &x, Cx4 &div)
{
  Dims3 const sz{x.dimension(0) - 2, x.dimension(1) - 2, x.dimension(2) - 2};
  Dims3 const st1{1, 1, 1};
  div.chip<3>(0).slice(st1, sz).device(Threads::GlobalDevice()) =
      fdiff(x.chip<3>(0), 0) + fdiff(x.chip<3>(3), 1) + fdiff(x.chip<3>(4), 2);
  div.chip<3>(1).slice(st1, sz).device(Threads::GlobalDevice()) =
      fdiff(x.chip<3>(3), 0) + fdiff(x.chip<3>(1), 1) + fdiff(x.chip<3>(5), 2);
  div.chip<3>(2).slice(st1, sz).device(Threads::GlobalDevice()) =
      fdiff(x.chip<3>(4), 0) + fdiff(x.chip<3>(5), 1) + fdiff(x.chip<3>(2), 2);
}

Cx3 tgv(
    Cx3 &data,
    Cx3::Dimensions const &dims,
    EncodeFunction const &encode,
    DecodeFunction const &decode,
    long const max_its,
    float const thresh,
    float const alpha,
    float const reduction,
    float const step_size,
    Log &log)
{
  // Allocate all memory
  Cx3 u(dims);
  decode(data, u);
  Cx3 check(data.dimensions());
  encode(u, check);

  float const scale = Norm(u);
  data = data / data.constant(scale);
  u = u / u.constant(scale);
  Cx3 u_(dims);
  u_ = u;
  Cx3 u_old(dims);
  u_old.setZero();
  Cx4 grad_u(dims[0], dims[1], dims[2], 3);
  grad_u.setZero();
  Cx4 p(dims[0], dims[1], dims[2], 3);
  p.setZero();
  Cx3 divp(dims);
  divp.setZero();
  Cx4 xi(dims[0], dims[1], dims[2], 3);
  xi.setZero();
  Cx4 xi_(dims[0], dims[1], dims[2], 3);
  xi_.setZero();
  Cx4 xi_old(dims[0], dims[1], dims[2], 3);
  xi_old.setZero();
  // Symmetric rank-2 tensor stored as xx yy zz (xy + yx) (xz + zx) (yz + zy)
  Cx4 grad_xi(dims[0], dims[1], dims[2], 6);
  grad_xi.setZero();
  Cx4 q(dims[0], dims[1], dims[2], 6);
  q.setZero();
  Cx4 divq(dims[0], dims[1], dims[2], 3);
  divq.setZero();

  Cx3 r(data.dimensions());
  r.setZero();
  Cx3 v(data.dimensions());
  v.setZero();
  Cx3 RtC_ubar(data.dimensions());
  RtC_ubar.setZero();
  Cx3 RtC_v(dims);
  RtC_v.setZero();

  float const alpha00 = alpha;
  float const alpha10 = alpha / 2.f;
  float const alpha01 = alpha00 * reduction;
  float const alpha11 = alpha10 * reduction;

  // Step lengths
  float const tau_p = 1.f / step_size;
  float const tau_d = 1.f / (step_size / 2.f);

  float const norm_u0 = Norm(u);

  log.info(FMT_STRING("Starting TGV Scale {} Initial Norm {}"), scale, norm_u0);

  for (auto ii = 0.f; ii < max_its; ii++) {
    // Regularisation factors
    float const prog = static_cast<float>(ii) / ((max_its == 1) ? 1. : (max_its - 1.f));
    float const alpha0 = std::exp(std::log(alpha01) * prog + std::log(alpha00) * (1.f - prog));
    float const alpha1 = std::exp(std::log(alpha11) * prog + std::log(alpha10) * (1.f - prog));

    // Save previous
    u_old.device(Threads::GlobalDevice()) = u;
    xi_old.device(Threads::GlobalDevice()) = xi;

    // Sample image in non-cartesian k-space
    encode(u_, RtC_ubar);
    r.device(Threads::GlobalDevice()) = RtC_ubar - data;
    v.device(Threads::GlobalDevice()) = (1.f / (1.f + tau_d)) * (v + tau_d * r);

    // Div P calculation and u update
    calc_scalar_grad(u_, grad_u);
    p.device(Threads::GlobalDevice()) = p - tau_d * (grad_u + xi_);
    auto absp = (p * p.conjugate()).real().sum(Sz1{3}).sqrt();
    auto projp = absp.unaryExpr([alpha1](float const &x) { return std::max(1.f, x / alpha1); });
    auto bp = projp.broadcast(Dims3{1, 1, 3}).reshape(Dims4{dims[0], dims[1], dims[2], 3});
    p.device(Threads::GlobalDevice()) = p / bp;
    calc_vector_div(p, divp);
    decode(v, RtC_v);
    u.device(Threads::GlobalDevice()) = u - tau_p * (divp + RtC_v);

    // Div q calculation and xi update
    calc_vector_grad(xi_, grad_xi);
    q.device(Threads::GlobalDevice()) = q - tau_d * grad_xi;
    auto q1 = q.slice(Dims4{0, 0, 0, 0}, Dims4{dims[0], dims[1], dims[2], 3});
    auto q2 = q.slice(Dims4{0, 0, 0, 3}, Dims4{dims[0], dims[1], dims[2], 3});
    auto q1sum = (q1 * q1.conjugate()).real().sum(Sz1{3});
    auto q2sum = (q2 * q2.conjugate()).real().sum(Sz1{3});
    auto absq = (q1sum + q2sum * 2.f).sqrt();
    auto projq = absq.unaryExpr([alpha0](float const &x) { return std::max(1.f, x / alpha0); });
    auto bq = projq.broadcast(Dims3{1, 1, 6}).reshape(Dims4{dims[0], dims[1], dims[2], 6});
    q.device(Threads::GlobalDevice()) = q / bq;
    calc_tensor_div(q, divq);
    xi.device(Threads::GlobalDevice()) = xi - tau_p * (divq - p);

    // Update bar variables
    u_.device(Threads::GlobalDevice()) = 2.0 * u - u_old;
    xi_.device(Threads::GlobalDevice()) = 2.0 * xi - xi_old;

    float const delta = Norm(u - u_old) / norm_u0;

    log.info(FMT_STRING("Iteration {}/{} alpha0 {} delta {}"), ii + 1, max_its, alpha0, delta);
    if (delta < thresh) {
      log.info("Reached threshold on delta, stopping");
      break;
    }
  }

  u = u * u.constant(scale);
  return u;
}