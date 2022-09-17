#include "dict.hpp"

#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

#include <numeric>

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

TreeNode::TreeNode(std::vector<Eigen::VectorXd> &points)
{
  if (points.size() == 1) {
    centroid = points[0];
  } else {
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(points.front().rows());
    centroid = std::accumulate(points.begin(), points.end(), zero);
    // centroid /= centroid.constant(points.size());
    centroid.normalize();

    Eigen::VectorXd leftCenter(centroid.rows()), rightCenter(centroid.rows());
    double worst = std::numeric_limits<double>::infinity();
    for (auto const &p : points) {
      double s = p.dot(centroid);
      if (s < worst) {
        worst = s;
        leftCenter = p;
      }
    }
    worst = std::numeric_limits<double>::infinity();
    for (auto const &p : points) {
      double s = p.dot(leftCenter);
      if (s < worst) {
        worst = s;
        rightCenter = p;
      }
    }
    if (leftCenter == rightCenter) {
      // Degenerate points, can't do better
      centroid = leftCenter;
    } else {
      std::vector<Eigen::VectorXd> leftPoints, rightPoints;
      for (auto const &p : points) {
        double const sLeft = p.dot(leftCenter);
        double const sRight = p.dot(rightCenter);
        if (sLeft < sRight) {
          rightPoints.push_back(p);
        } else {
          leftPoints.push_back(p);
        }
      }
      left = std::make_unique<TreeNode>(leftPoints);
      right = std::make_unique<TreeNode>(rightPoints);
    }
  }
}

auto TreeNode::find(Eigen::VectorXcd const &p) const -> Eigen::VectorXd
{
  if (left) {
    double leftDist = std::abs(p.dot(left->centroid.cast<Cxd>()));
    double rightDist = std::abs(p.dot(right->centroid.cast<Cxd>()));
    if (leftDist < rightDist) {
      return right->find(p);
    } else {
      return left->find(p);
    }
  } else {
    return centroid;
  }
}

TreeProjection::TreeProjection(Re2 d)
  : Functor<Cx4>()
{
  std::vector<Eigen::VectorXd> points;
  for (Index ii = 0; ii < d.dimension(0); ii++) {
    Eigen::VectorXd temp(d.dimension(1));
    for (Index ij = 0; ij < d.dimension(1); ij++) {
      temp(ij) = d(ii, ij);
    }
    points.push_back(temp.normalized());
  }
  Log::Print("Building ball-tree");
  root = std::make_unique<TreeNode>(points);
  Log::Print("Finished building tree");
}

auto TreeProjection::operator()(Cx4 const &x) const -> Cx4
{
  Cx4 y(x.dimensions());
  auto ztask = [&](Index const iz) {
    for (Index iy = 0; iy < x.dimension(2); iy++) {
      for (Index ix = 0; ix < x.dimension(1); ix++) {
        Cx1 const p = x.chip<3>(iz).chip<2>(iy).chip<1>(ix);
        Eigen::VectorXcd pv(x.dimension(0));
        for (Index ii = 0; ii < x.dimension(0); ii++) {
          pv(ii) = p(ii);
        }
        Eigen::VectorXcd const d = root->find(pv).cast<Cxd>();
        std::complex rho = pv.dot(d);
        for (Index ii = 0; ii < x.dimension(0); ii++) {
          y(ii, ix, iy, iz) = rho * d(ii);
        }
      }
    }
  };
  Threads::For(ztask, x.dimension(3), "Dictionary Projection");
  return y;
}

} // namespace rl