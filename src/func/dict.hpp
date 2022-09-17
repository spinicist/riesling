#pragma once

#include "functor.hpp"
#include <memory>

namespace rl {

struct DictionaryProjection final : Functor<Cx4>
{
  Re2 dictionary = Re2();
  DictionaryProjection(Re2);

  auto operator()(const Cx4 &) const -> Cx4;
};

struct TreeNode {
  TreeNode(std::vector<Eigen::VectorXd> &points);
  auto find(Eigen::VectorXcd const &p) const -> Eigen::VectorXd;
  Eigen::VectorXd centroid;
  std::unique_ptr<TreeNode> left = nullptr, right = nullptr;
};

struct TreeProjection final : Functor<Cx4>
{
  TreeProjection(Re2);
  std::unique_ptr<TreeNode> root;

  auto operator()(const Cx4 &) const -> Cx4;
};

} // namespace rl
