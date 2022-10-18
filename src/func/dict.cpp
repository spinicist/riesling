#include "dict.hpp"

#include "log.hpp"
#include "tensorOps.hpp"
#include "threads.hpp"

#include <numeric>

namespace rl {

auto LookupDictionary::operator()(Eigen::TensorMap<Cx4 const>x) const -> Eigen::TensorMap<Cx4>
{
  Log::Print("Dictionary projection. Dims {}", x.dimensions());
  static Cx4 y(x.dimensions());
  auto ztask = [&](Index const iz) {
    Eigen::VectorXcf pv(x.dimension(0));
    for (Index iy = 0; iy < x.dimension(2); iy++) {
      for (Index ix = 0; ix < x.dimension(1); ix++) {
        for (Index ii = 0; ii < x.dimension(0); ii++) {
          pv(ii) = x(ii, ix, iy, iz);
        }
        Eigen::VectorXcf const dv = this->project(pv);
        for (Index ii = 0; ii < x.dimension(0); ii++) {
          y(ii, ix, iy, iz) = dv(ii);
        }
      }
    }
  };
  Threads::For(ztask, x.dimension(3), "Dictionary Projection");
  return y;
}

BruteForceDictionary::BruteForceDictionary(Eigen::MatrixXf const &d)
  : LookupDictionary()
  , dictionary{d}
{
  Log::Print("Brute-Force Dictionary rows {} entries {}", d.rows(), d.cols());
}

auto BruteForceDictionary::project(Eigen::VectorXcf const &p) const -> Eigen::VectorXcf
{
  Cx bestρ{0.f, 0.f};
  Index bestIndex = -1;
  for (Index ii = 0; ii < dictionary.cols(); ii++) {
    Cx const ρ = dictionary.col(ii).cast<Cx>().dot(p);
    if (std::abs(ρ) > std::abs(bestρ)) {
      bestρ = ρ;
      bestIndex = ii;
    }
  }
  // fmt::print(FMT_STRING("bestρ {} p {} d {} proj {}\n"), bestρ, p.transpose(), dictionary.col(bestIndex).transpose(), (dictionary.col(bestIndex) * bestρ).transpose());
  return dictionary.col(bestIndex) * bestρ;
}

TreeNode::TreeNode(std::vector<Eigen::VectorXf> &points)
{
  if (points.size() == 1) {
    centroid = points[0];
  } else if (points.size() == 2) {
    // Skip directly to inserting two separate nodes into the tree
    centroid = (points.front() + points.back()) / 2.f;
    std::vector<Eigen::VectorXf> leftPoints, rightPoints;
    leftPoints.push_back(points.front());
    rightPoints.push_back(points.back());
    left = std::make_unique<TreeNode>(leftPoints);
    right = std::make_unique<TreeNode>(rightPoints);
  } else {
    Eigen::VectorXf zero = Eigen::VectorXf::Zero(points.front().rows());
    centroid = std::accumulate(points.begin(), points.end(), zero);
    centroid /= points.size();

    Eigen::VectorXf leftCenter(centroid.rows()), rightCenter(centroid.rows());
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

    std::vector<Eigen::VectorXf> leftPoints, rightPoints;
    for (auto const &p : points) {
      double const sLeft = p.dot(leftCenter);
      double const sRight = p.dot(rightCenter);
      if (sLeft < sRight) {
        rightPoints.push_back(p);
      } else {
        leftPoints.push_back(p);
      }
    }

    if (leftPoints.size() && rightPoints.size()) {
      left = std::make_unique<TreeNode>(leftPoints);
      right = std::make_unique<TreeNode>(rightPoints);
    } else {
      // Maths broke
      // fmt::print("Maths broke\n");
    }
  }
}

auto TreeNode::find(Eigen::VectorXcf const &p) const -> Eigen::VectorXf
{
  if (left) {
    double leftDist = std::abs(left->centroid.cast<Cx>().dot(p));
    double rightDist = std::abs(right->centroid.cast<Cx>().dot(p));
    if (leftDist < rightDist) {
      return right->find(p);
    } else {
      return left->find(p);
    }
  } else {
    return centroid;
  }
}

BallTreeDictionary::BallTreeDictionary(Eigen::MatrixXf const &dict)
  : LookupDictionary()
{
  std::vector<Eigen::VectorXf> points;
  Eigen::VectorXf temp(dict.rows());
  for (Index ii = 0; ii < dict.cols(); ii++) {
    for (Index ij = 0; ij < dict.rows(); ij++) {
      temp(ij) = dict(ij, ii);
    }
    points.push_back(temp.normalized());
  }
  Log::Print("Building Ball-Tree Dictionary rows {} entries {}", dict.rows(), dict.cols());
  root = std::make_unique<TreeNode>(points);
  Log::Print("Finished building tree");
}

auto BallTreeDictionary::project(Eigen::VectorXcf const &p) const -> Eigen::VectorXcf
{
  auto const d = root->find(p);
  return d * d.cast<Cx>().dot(p);
}

} // namespace rl