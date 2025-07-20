# Installation

## Install

# Usage

## Training

1. Put ShapeNet dataset in data/ShapeNetCore
2. Preprocess dataset using generate_targets.py, obj_to_cloud.py and redistribute_shapenet.py. You can also use already generated targets in plane_cache.json
3. Run train.py. Change the constants inside the script to configure training.

## Inference

To decompose a mesh use:

> python decompose.py _filename_ _depth_(optional)

For evaluation, place v-hacd dataset in data/v_hacd and use evaluate_coacd.py or evaluate_neuralacd.py. Change RANDOM_ROTATION to True to use random rotation.
