#pragma once

#include <mesh.hpp>

namespace acd_gen {

void SDFManifold(Mesh &input, Mesh &output, double scale = 50.0f,
                 double level_set = 0.55f);
void ManifoldPreprocess(Mesh &m, double scale = 50.0f,
                        double level_set = 0.55f);
} // namespace acd_gen