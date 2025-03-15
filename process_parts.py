from utils.preprocessor import preprocess_data
from utils.BaseUtils import *
import json

import coacd_modified

coacd_modified.set_log_level("off")

if __name__ == "__main__":
    loader = load_shapenet(data_folder="data/ShapeNetParts")
    with open("plane_cache.json", "r") as f:
        plane_cache = json.load(f)
    preprocessor = preprocess_data(loader=loader,plane_cache=plane_cache)

    while True:
        next(preprocessor)

