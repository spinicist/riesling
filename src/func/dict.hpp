#pragma once

#include "functor.hpp"
#include <memory>

namespace rl {

struct LookupDictionary : Functor<Cx4> {
  using Parent = Functor<Cx4>;
  using typename Parent::Input;
  using typename Parent::Output;

  void operator()(Input x, Output y) const;
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
  std::shared_ptr<TreeNode> left = nullptr, right = nullptr;
};

struct BallTreeDictionary final : LookupDictionary
{
  BallTreeDictionary(Eigen::MatrixXf const &d);
  std::shared_ptr<TreeNode> root;

  auto project(Eigen::VectorXcf const &p) const -> Eigen::VectorXcf;
};

} // namespace rl
