This repository contains code for both coarse (place recognition folder) and fine localization (multi-constellation folder).

The place_recognition folder includes:
- Source code for proposed descriptor extraction and retrieval.
- C++ implementations for some common descriptors.

The multi-constellation folder includes:
- scancontext folder -> multi-constellation folder.
- circle3.cpp -> Solving the system of equations for multiple spheres.
- datasource.cpp -> Point Cloud and GPS Data Input.
- descriptor.cpp -> Loading or saving the generated descriptors.
- generate_desc.cpp -> Generate descriptors, avoiding duplicate calculations.
- liosam_trace_to_txt.cpp -> Convert LIOSAM trajectory to the required format. (Other mapping methods are also acceptable, as long as the output trajectory format is identical.)
- loam.hpp -> Lightweight LOAM for rapid, repeated registration.
- registration.cpp -> Baseline method implementation (ICP, NDT, and et al.)
- trace2.cpp -> The core C++ code of the experiment.
- CMakeLists.txt -> It is used for compiling the entire project.

The self-collected data is available at [https://www.kaggle.com/datasets/anacondaspyder/self-collected-dataset.](https://www.kaggle.com/datasets/anacondaspyder/self-collected-dataset?select=154843)
- In our experiment, the mapping data consists of frames sampled at 0.4-second intervals, with the remaining frames serving as the localization data.
- Other allocation schemes are also acceptable, as long as the mapping and localization data are distinct.
  
