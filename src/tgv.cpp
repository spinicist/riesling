#include "tgv.h"

#include "tensorOps.h"
#include "threads.h"

inline auto ForwardDiff(Cx3 const &a, Eigen::Index const d)
{
  Sz3 const sz{a.dimension(0) - 2, a.dimension(1) - 2, a.dimension(2) - 2};
  Sz3 const st1{1, 1, 1};
  Sz3 fwd{1, 1, 1};
  fwd[d] = 2;

  return (a.slice(fwd, sz) - a.slice(st1, sz));
}

inline auto BackwardDiff(Cx3 const &a, Eigen::Index const d)
{
  Sz3 const sz{a.dimension(0) - 2, a.dimension(1) - 2, a.dimension(2) - 2};
  Sz3 const st1{1, 1, 1};
  Sz3 bck{1, 1, 1};
  bck[d] = 0;

  return (a.slice(st1, sz) - a.slice(bck, sz));
}

inline auto CentralDiff(Cx3 const &a, Eigen::Index const d)
{
  Sz3 const sz{a.dimension(0) - 2, a.dimension(1) - 2, a.dimension(2) - 2};
  Sz3 const st1{1, 1, 1};
  Sz3 fwd{1, 1, 1};
  Sz3 bck{1, 1, 1};
  fwd[d] = 2;
  bck[d] = 0;

  return (a.slice(fwd, sz) - a.slice(bck, sz)) / a.slice(st1, sz).constant(2.f);
}

inline void Grad(Cx3 const &a, Cx4 &g, Eigen::ThreadPoolDevice &dev)
{
  Sz3 const sz{a.dimension(0) - 2, a.dimension(1) - 2, a.dimension(2) - 2};
  Sz3 const st1{1, 1, 1};
  g.chip<3>(0).slice(st1, sz).device(dev) = ForwardDiff(a, 0);
  g.chip<3>(1).slice(st1, sz).device(dev) = ForwardDiff(a, 1);
  g.chip<3>(2).slice(st1, sz).device(dev) = ForwardDiff(a, 2);
}

inline void Grad(Cx4 const &x, Cx4 &gx, Eigen::ThreadPoolDevice &dev)
{
  Sz3 const sz{x.dimension(0) - 2, x.dimension(1) - 2, x.dimension(2) - 2};
  Sz3 const st1{1, 1, 1};

  gx.chip<3>(0).slice(st1, sz).device(dev) = BackwardDiff(x.chip<3>(0), 0);
  gx.chip<3>(1).slice(st1, sz).device(dev) = BackwardDiff(x.chip<3>(1), 1);
  gx.chip<3>(2).slice(st1, sz).device(dev) = BackwardDiff(x.chip<3>(2), 2);

  gx.chip<3>(3).slice(st1, sz).device(dev) =
    (BackwardDiff(x.chip<3>(0), 1) + BackwardDiff(x.chip<3>(1), 0)) /
    gx.chip<3>(3).slice(st1, sz).constant(2.f);

  gx.chip<3>(4).slice(st1, sz).device(dev) =
    (BackwardDiff(x.chip<3>(0), 2) + BackwardDiff(x.chip<3>(2), 0)) /
    gx.chip<3>(4).slice(st1, sz).constant(2.f);

  gx.chip<3>(5).slice(st1, sz).device(dev) =
    (BackwardDiff(x.chip<3>(1), 2) + BackwardDiff(x.chip<3>(2), 1)) /
    gx.chip<3>(5).slice(st1, sz).constant(2.f);
}

inline void Div(Cx4 const &x, Cx3 &div, Eigen::ThreadPoolDevice &dev)
{
  Sz3 const sz{x.dimension(0) - 2, x.dimension(1) - 2, x.dimension(2) - 2};
  Sz3 const st1{1, 1, 1};
  div.slice(st1, sz).device(dev) =
    BackwardDiff(x.chip<3>(0), 0) + BackwardDiff(x.chip<3>(1), 1) + BackwardDiff(x.chip<3>(2), 2);
}

inline void Div(Cx4 const &x, Cx4 &div, Eigen::ThreadPoolDevice &dev)
{
  Sz3 const sz{x.dimension(0) - 2, x.dimension(1) - 2, x.dimension(2) - 2};
  Sz3 const st1{1, 1, 1};
  div.chip<3>(0).slice(st1, sz).device(dev) =
    ForwardDiff(x.chip<3>(0), 0) + ForwardDiff(x.chip<3>(3), 1) + ForwardDiff(x.chip<3>(4), 2);
  div.chip<3>(1).slice(st1, sz).device(dev) =
    ForwardDiff(x.chip<3>(3), 0) + ForwardDiff(x.chip<3>(1), 1) + ForwardDiff(x.chip<3>(5), 2);
  div.chip<3>(2).slice(st1, sz).device(dev) =
    ForwardDiff(x.chip<3>(4), 0) + ForwardDiff(x.chip<3>(5), 1) + ForwardDiff(x.chip<3>(2), 2);
}

inline void ProjectP(Cx4 &p, float const a, Eigen::ThreadPoolDevice &dev)
{
  Eigen::IndexList<int, int, int, Eigen::type2index<1>> res;
  res.set(0, p.dimension(0));
  res.set(1, p.dimension(1));
  res.set(2, p.dimension(2));
  Eigen::IndexList<
    Eigen::type2index<1>,
    Eigen::type2index<1>,
    Eigen::type2index<1>,
    Eigen::type2index<3>>
    brd;

  R3 normp(p.dimension(0), p.dimension(1), p.dimension(2));
  normp.device(dev) = (p * p.conjugate()).sum(Sz1{3}).real().sqrt() / a;
  normp.device(dev) = (normp > 1.f).select(normp, normp.constant(1.f));
  p.device(dev) = p / normp.reshape(res).broadcast(brd).cast<Cx>();
}

inline void ProjectQ(Cx4 &q, float const a, Eigen::ThreadPoolDevice &dev)
{
  Eigen::IndexList<int, int, int, Eigen::type2index<1>> res;
  res.set(0, q.dimension(0));
  res.set(1, q.dimension(1));
  res.set(2, q.dimension(2));
  Eigen::IndexList<
    Eigen::type2index<1>,
    Eigen::type2index<1>,
    Eigen::type2index<1>,
    Eigen::type2index<6>>
    brd;

  auto const qsqr = q * q.conjugate();
  auto const q1 =
    qsqr.slice(Sz4{0, 0, 0, 0}, Sz4{q.dimension(0), q.dimension(1), q.dimension(2), 3});
  auto const q2 =
    qsqr.slice(Sz4{0, 0, 0, 3}, Sz4{q.dimension(0), q.dimension(1), q.dimension(2), 3});
  R3 normq(q.dimension(0), q.dimension(1), q.dimension(2));
  normq.device(dev) = (q1.sum(Sz1{3}).real() + q2.sum(Sz1{3}).real() * 2.f).sqrt() / a;
  normq.device(dev) = (normq > 1.f).select(normq, normq.constant(1.f));
  q.device(dev) = q / normq.reshape(res).broadcast(brd).cast<Cx>();
}

Cx3 tgv(
  long const max_its,
  float const thresh,
  float const alpha,
  float const reduction,
  float const step_size,
  ReconOp &op,
  Cx3 const &ks_data,
  Log &log)
{
  auto dev = Threads::GlobalDevice();

  auto const dims = op.dimensions();
  Sz4 dims3{dims[0], dims[1], dims[2], 3};
  Sz4 dims6{dims[0], dims[1], dims[2], 6};

  // Primal variables
  Cx3 u(dims);                 // Main variable
  op.Adj(ks_data, u);          // Get starting point
  float const scale = Norm(u); // Normalise regularisation factors
  Cx3 u_ = u;                  // Bar variable (is this the "dual"?)
  Cx3 u_old = u;               // From previous iteration
  Cx4 grad_u(dims3);
  grad_u.setZero();
  Cx4 v(dims3);
  v.setZero();
  Cx4 v_(dims3);
  v_.setZero();
  Cx4 v_old(dims3);
  v_old.setZero();
  Cx4 grad_v(dims6); // Symmetric rank-2 tensor stored as xx yy zz (xy + yx) (xz + zx) (yz + zy)
  grad_v.setZero();

  // Dual variables
  Cx4 p(dims3);
  p.setZero();
  Cx3 divp(dims);
  divp.setZero();
  Cx4 q(dims6);
  q.setZero();
  Cx4 divq(dims3);
  divq.setZero();
  Cx3 v_decode(dims);
  v_decode.setZero();

  // k-Space variables
  Cx3 ks_res(ks_data.dimensions()); // Residual term in k-space
  ks_res.setZero();
  Cx3 r(ks_data.dimensions());
  r.setZero();

  float const alpha00 = scale * alpha;
  float const alpha10 = scale * alpha / 2.f;
  float const alpha01 = alpha00 * reduction;
  float const alpha11 = alpha10 * reduction;

  // Step lengths
  float const tau_p = 1.f / step_size;
  float const tau_d = 1.f / (step_size / 2.f);

  log.info(FMT_STRING("TGV Scale {}"), scale);

  for (auto ii = 0.f; ii < max_its; ii++) {
    log.image(u, fmt::format(FMT_STRING("tgv-u-{:02}.nii"), ii));
    // Regularisation factors
    float const prog = static_cast<float>(ii) / ((max_its == 1) ? 1. : (max_its - 1.f));
    float const alpha0 = std::exp(std::log(alpha01) * prog + std::log(alpha00) * (1.f - prog));
    float const alpha1 = std::exp(std::log(alpha11) * prog + std::log(alpha10) * (1.f - prog));

    // Update p
    Grad(u_, grad_u, dev);
    p.device(dev) = p - tau_d * (grad_u + v_);
    ProjectP(p, alpha1, dev);

    // Update q
    Grad(v_, grad_v, dev);
    q.device(dev) = q - tau_d * grad_v;
    ProjectQ(q, alpha0, dev);

    // Update r (in k-space)
    op.A(u_, ks_res);
    ks_res.device(dev) = ks_res - ks_data;
    r.device(dev) = (r + tau_d * ks_res) / r.constant(1.f + tau_d); // Prox op

    // Update u
    u_old.device(dev) = u;
    Div(p, divp, dev);
    op.Adj(r, v_decode);
    u.device(dev) = u - tau_p * (divp + v_decode); // Paper says +tau, but code says -tau
    u_.device(dev) = 2.0 * u - u_old;

    // Update v
    v_old.device(dev) = v;
    Div(q, divq, dev);
    v.device(dev) = v - tau_p * (divq - p);
    v_.device(dev) = 2.0 * v - v_old;

    // Check for convergence
    float const delta = Norm(u - u_old);
    log.info(FMT_STRING("TGV {}: ɑ0 {} δ {}"), ii + 1, alpha0, delta);
    if (delta < thresh) {
      log.info("Reached threshold on delta, stopping");
      break;
    }
  };
  return u;
}