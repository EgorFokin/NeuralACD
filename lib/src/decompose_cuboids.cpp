#include <algorithm>
#include <cmath>
#include <core.hpp>
#include <cuboid.hpp>
#include <stdexcept>
#include <vector>

namespace neural_acd {

std::vector<Cuboid> handle_collisions2D(Cuboid &part1, Cuboid &part2, int d1,
                                        int d2, double eps = 1e-6) {
  // first element of the returned vector should be colliding part
  // second element is the untouched part

  // comments for xy
  if (part1.min[d1] > eps + part2.min[d1] &&
      part1.max[d1] + eps < part2.max[d1]) {
    // Part 1x in Part 2x
    if (!(part1.min[d2] > eps + part2.min[d2] &&
          part1.max[d2] + eps < part2.max[d2])) {
      // Part 1y and Part 2y collide
      if (part1.min[d2] > eps + part2.min[d2]) {
        // part 1y is above Part 2y-min
        Cuboid colliding_part = part1;
        colliding_part.update_side("+", d2, part2.max[d2]);
        part1.update_side("-", d2, part2.max[d2]);
        return {colliding_part, part2, part1};
      } else if (part1.max[d2] + eps < part2.max[d2]) {
        // part 1y is below Part 2y-max
        Cuboid colliding_part = part1;
        colliding_part.update_side("-", d2, part2.min[d2]);
        part1.update_side("+", d2, part2.min[d2]);
        return {colliding_part, part2, part1};
      } else {
        Cuboid new_part = part1, colliding_part = part1;
        colliding_part.update_side("+", d2, part2.max[d2]);
        colliding_part.update_side("-", d2, part2.min[d2]);
        new_part.update_side("-", d2, part2.max[d2]);
        part1.update_side("+", d2, part2.min[d2]);
        return {colliding_part, part2, part1, new_part};
      }
    }
  } else if (part2.min[d1] > eps + part1.min[d1] &&
             part2.max[d1] + eps < part1.max[d1]) {
    // Part 2x in Part 1x, do the same thing
    if (!(part2.min[d2] > eps + part1.min[d2] &&
          part2.max[d2] + eps < part1.max[d2])) {
      if (part2.min[d2] > eps + part1.min[d2]) {
        Cuboid colliding_part = part2;
        colliding_part.update_side("+", d2, part1.max[d2]);
        part2.update_side("-", d2, part1.max[d2]);
        return {colliding_part, part1, part2};
      } else if (part2.max[d2] + eps < part1.max[d2]) {
        Cuboid colliding_part = part2;
        colliding_part.update_side("-", d2, part1.min[d2]);
        part2.update_side("+", d2, part1.min[d2]);
        return {colliding_part, part1, part2};
      } else {
        Cuboid new_part = part2, colliding_part = part2;
        colliding_part.update_side("+", d2, part1.max[d2]);
        colliding_part.update_side("-", d2, part1.min[d2]);
        new_part.update_side("-", d2, part1.max[d2]);
        part2.update_side("+", d2, part1.min[d2]);
        return {colliding_part, part1, part2, new_part};
      }
    }
  } else if (part1.min[d2] > eps + part2.min[d2] &&
             part1.max[d2] + eps < part2.max[d2]) {
    // Part 1y in Part 2y
    // they are definetely not one inside another due to previous checks
    if (part1.min[d1] > eps + part2.min[d1]) {
      Cuboid colliding_part = part1;
      colliding_part.update_side("+", d1, part2.max[d1]);
      part1.update_side("-", d1, part2.max[d1]);
      return {colliding_part, part2, part1};
    } else if (part1.max[d1] + eps < part2.max[d1]) {
      Cuboid colliding_part = part1;
      colliding_part.update_side("-", d1, part2.min[d1]);
      part1.update_side("+", d1, part2.min[d1]);
      return {colliding_part, part2, part1};
    } else {
      Cuboid new_part = part1, colliding_part = part1;
      colliding_part.update_side("+", d1, part2.max[d1]);
      colliding_part.update_side("-", d1, part2.min[d1]);
      new_part.update_side("-", d1, part2.max[d1]);
      part1.update_side("+", d1, part2.min[d1]);
      return {colliding_part, part2, part1, new_part};
    }
  } else if (part2.min[d2] > eps + part1.min[d2] &&
             part2.max[d2] + eps < part1.max[d2]) {
    // Part 2y in Part 1y
    if (part2.min[d1] > eps + part1.min[d1]) {
      Cuboid colliding_part = part2;
      colliding_part.update_side("+", d1, part1.max[d1]);
      part2.update_side("-", d1, part1.max[d1]);
      return {colliding_part, part1, part2};
    } else if (part2.max[d1] + eps < part1.max[d1]) {
      Cuboid colliding_part = part2;
      colliding_part.update_side("-", d1, part1.min[d1]);
      part2.update_side("+", d1, part1.min[d1]);
      return {colliding_part, part1, part2};
    } else {
      Cuboid new_part = part2, colliding_part = part2;
      colliding_part.update_side("+", d1, part1.max[d1]);
      colliding_part.update_side("-", d1, part1.min[d1]);
      new_part.update_side("-", d1, part1.max[d1]);
      part2.update_side("+", d1, part1.min[d1]);
      return {colliding_part, part1, part2, new_part};
    }
  } else {
    // corners are colliding

    if (part1.min[d1] > eps + part2.min[d1] &&
        part1.min[d2] > eps + part2.min[d2]) {
      // top right of part2
      Cuboid new_part = part1, colliding_part = part1;
      colliding_part.update_side("+", d2, part2.max[d2]);
      colliding_part.update_side("+", d1, part2.max[d1]);

      new_part.update_side("-", d2, part2.max[d2]);
      new_part.update_side("+", d1, part2.max[d1]);

      part1.update_side("-", d1, part2.max[d1]);
      return {colliding_part, part2, part1, new_part};
    } else if (part1.min[d1] > eps + part2.min[d1] &&
               part1.max[d2] + eps < part2.max[d2]) {
      // bottom right of part2
      Cuboid new_part = part1, colliding_part = part1;
      colliding_part.update_side("-", d2, part2.min[d2]);
      colliding_part.update_side("+", d1, part2.max[d1]);

      new_part.update_side("+", d2, part2.min[d2]);
      new_part.update_side("+", d1, part2.max[d1]);

      part1.update_side("-", d1, part2.max[d1]);
      return {colliding_part, part2, part1, new_part};
    } else if (part1.max[d1] + eps < part2.max[d1] &&
               part1.min[d2] > eps + part2.min[d2]) {
      // top left of part2
      Cuboid new_part = part1, colliding_part = part1;
      colliding_part.update_side("+", d2, part2.max[d2]);
      colliding_part.update_side("-", d1, part2.min[d1]);

      new_part.update_side("-", d2, part2.max[d2]);
      new_part.update_side("-", d1, part2.min[d1]);

      part1.update_side("+", d1, part2.min[d1]);
      return {colliding_part, part2, part1, new_part};
    } else if (part1.max[d1] + eps < part2.max[d1] &&
               part1.max[d2] + eps < part2.max[d2]) {
      // bottom left of part2
      Cuboid new_part = part1, colliding_part = part1;
      colliding_part.update_side("-", d2, part2.min[d2]);
      colliding_part.update_side("-", d1, part2.min[d1]);

      new_part.update_side("+", d2, part2.min[d2]);
      new_part.update_side("-", d1, part2.min[d1]);

      part1.update_side("+", d1, part2.min[d1]);
      return {colliding_part, part2, part1, new_part};
    }
  }

  if (part1.min[d1] > eps + part2.min[d1] &&
      part1.max[d1] + eps < part2.max[d1] &&
      part1.min[d2] > eps + part2.min[d2] &&
      part1.max[d2] + eps < part2.max[d2]) {
    // Part 1 is completely inside Part 2
    return {part1, part2};
  } else if (part2.min[d1] > eps + part1.min[d1] &&
             part2.max[d1] + eps < part1.max[d1] &&
             part2.min[d2] > eps + part1.min[d2] &&
             part2.max[d2] + eps < part1.max[d2]) {
    // Part 2 is completely inside Part 1
    return {part2, part1};
  }

  return {part1, part2}; // probably the same from that dimension
}

std::vector<Cuboid> handle_two_colliding_parts(Cuboid &part1, Cuboid &part2) {
  // Handle collision between two parts in 3D space
  std::vector<Cuboid> new_parts;
  Cuboid *p1 = &part1;
  Cuboid *p2 = &part2;

  // Check for collisions in each pair of dimensions
  auto xy_parts = handle_collisions2D(*p1, *p2, 0, 1);

  if (xy_parts.size() >= 3) {
    new_parts.insert(new_parts.end(), xy_parts.begin() + 2, xy_parts.end());
    p1 = &xy_parts[0];
    p2 = &xy_parts[1];
  }

  auto xz_parts = handle_collisions2D(*p1, *p2, 0, 2);

  if (xz_parts.size() >= 3) {
    new_parts.insert(new_parts.end(), xz_parts.begin() + 2, xz_parts.end());
    p1 = &xz_parts[0];
    p2 = &xz_parts[1];
  }

  auto yz_parts = handle_collisions2D(*p1, *p2, 1, 2);

  if (yz_parts.size() >= 3) {
    new_parts.insert(new_parts.end(), yz_parts.begin() + 2, yz_parts.end());
    p1 = &yz_parts[0];
    p2 = &yz_parts[1];
  }

  // new_parts.push_back(*p1);
  new_parts.push_back(*p2);

  return new_parts;
}

Cuboid *get_colliding_part(std::vector<Cuboid> &parts, Cuboid &new_part) {
  // Check for collisions with existing parts
  Cuboid *colliding_part = nullptr;
  for (auto &part : parts) {
    if (check_aabb_collision(part, new_part)) {
      colliding_part = &part;
      break;
    }
  }
  return colliding_part; // No collision detected
}

void remove_flat(std::vector<Cuboid> &parts, double threshold = 0.05) {
  parts.erase(std::remove_if(parts.begin(), parts.end(),
                             [threshold](const Cuboid &part) {
                               return (part.max[0] - part.min[0] < threshold ||
                                       part.max[1] - part.min[1] < threshold ||
                                       part.max[2] - part.min[2] < threshold);
                             }),
              parts.end());
}

void merge_adjacent_cuboids(std::vector<Cuboid> &parts, double eps = 1e-6) {
  for (size_t i = 0; i < parts.size(); ++i) {
    for (size_t j = parts.size() - 1; j > i; --j) {
      int same_sides = 0;
      same_sides += (std::abs(parts[i].min[0] - parts[j].min[0]) < eps);
      same_sides += (std::abs(parts[i].max[0] - parts[j].max[0]) < eps);
      same_sides += (std::abs(parts[i].min[1] - parts[j].min[1]) < eps);
      same_sides += (std::abs(parts[i].max[1] - parts[j].max[1]) < eps);
      same_sides += (std::abs(parts[i].min[2] - parts[j].min[2]) < eps);
      same_sides += (std::abs(parts[i].max[2] - parts[j].max[2]) < eps);

      if (same_sides == 4) {
        if (std::abs(parts[i].min[0] - parts[j].max[0]) < eps) {
          // Merge along x-axis
          parts[i].update_side("-", 0, parts[j].min[0]);
        } else if (std::abs(parts[i].max[0] - parts[j].min[0]) < eps) {
          // Merge along x-axis
          parts[i].update_side("+", 0, parts[j].max[0]);
        } else if (std::abs(parts[i].min[1] - parts[j].max[1]) < eps) {
          // Merge along y-axis
          parts[i].update_side("-", 1, parts[j].min[1]);
        } else if (std::abs(parts[i].max[1] - parts[j].min[1]) < eps) {
          // Merge along y-axis
          parts[i].update_side("+", 1, parts[j].max[1]);
        } else if (std::abs(parts[i].min[2] - parts[j].max[2]) < eps) {
          // Merge along z-axis
          parts[i].update_side("-", 2, parts[j].min[2]);
        } else if (std::abs(parts[i].max[2] - parts[j].min[2]) < eps) {
          // Merge along z-axis
          parts[i].update_side("+", 2, parts[j].max[2]);
        } else
          continue;

        // Remove the merged part
        parts.erase(parts.begin() + j);
        merge_adjacent_cuboids(parts);
        return; // Restart merging from the beginning
      }
    }
  }
}

void update_decomposition_step(std::vector<Cuboid> &parts, Cuboid &new_part) {
  Cuboid *colliding_part = get_colliding_part(parts, new_part);

  if (colliding_part == nullptr) {
    // If no collisions, add the new part to the decomposition
    parts.push_back(new_part);
    return;
  }

  std::vector<Cuboid> new_parts =
      handle_two_colliding_parts(*colliding_part, new_part);

  remove_flat(new_parts);

  // erase the original part from the parts vector
  parts.erase(parts.begin() + (colliding_part - parts.data()));

  for (auto &part : new_parts)
    update_decomposition_step(parts, part);
}

void update_decomposition(std::vector<Cuboid> &parts, Cuboid &new_part) {
  update_decomposition_step(parts, new_part);
}

} // namespace neural_acd