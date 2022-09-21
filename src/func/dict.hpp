#pragma once

#include "functor.hpp"
#include <memory>

namespace rl {

struct LookupDictionary : Functor<Cx4> {
  auto operator()(const Cx4 &) const -> Cx4;
  virtual auto project(Eigen::VectorXcf const &p) const -> Eigen::VectorXcf = 0;
};

struct BruteForceDictionary final : LookupDictionary
{
  Eigen::MatrixXf dictionary;
  BruteForceDictionary(Eigen::MatrixXf const &d);

  auto project(Eigen::VectorXcf const &p) const -> Eigen::VectorXcf;
};

struct TreeNode {
  TreeNode(std::vector<Eigen::VectorXf> &points);
  auto find(Eigen::VectorXcf const &p) const -> Eigen::VectorXf;
  Eigen::VectorXf centroid;
  std::unique_ptr<TreeNode> left = nullptr, right = nullptr;
};

struct BallTreeDictionary final : LookupDictionary
{
  BallTreeDictionary(Eigen::MatrixXf const &d);
  std::unique_ptr<TreeNode> root;

  auto project(Eigen::VectorXcf const &p) const -> Eigen::VectorXcf;
};

} // namespace rl
