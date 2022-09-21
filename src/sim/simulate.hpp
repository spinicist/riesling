#pragma once

namespace riesling {

struct Simulate(rl::Settings const &s, Index const nsamp)
{
  T simulator{s};

  Eigen::ArrayXXf parameters = simulator.parameters(nsamp);
  Eigen::ArrayXXf dynamics(parameters.cols(), simulator.length());
  auto const start = Log::Now();
  auto task = [&](Index const ii) { dynamics.row(ii) = simulator.simulate(parameters.col(ii)); };
  Threads::For(task, parameters.cols(), "Simulation");
  Log::Print(FMT_STRING("Simulation took {}"), Log::ToNow(start));
  return std::make_tuple(parameters, dynamics);
}

}