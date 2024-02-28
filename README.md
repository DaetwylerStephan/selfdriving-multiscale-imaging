# self-driving, multi-scale imaging
This repository contains the tools to process and analyse the multiscale data.



### morphological_feature_analysis
Contains code to run the morphological feature analysis in MATLAB of the high-resolution macrophages.

### multiScale_hardware_control
Contains code to run a self-driving, multi-scale light-sheet microscope, including the context_driven_env.yml environment file.

### Segmentation-highres
Contains code to run our high-resolution macrophage segmentation pipeline using connected component labeling and a Cellpose-based 3D segmentation pipeline. The environmen is available as .yml file.

### Segmentation_lowres
Contains code to run our low-resolution macrophage segmentation pipeline based on CLIJ, including a .yml environment file. 

### Stitching
Contains code to stitch 3D data using translation based on the stage positions using Fast Fourier transform-based cross-correlations on the overlapping sections.

### Visualization
Contains code to visualize time-lapse data using ImageJ (generatelabel_colormap.ijm) and Napari (low- and high-res data).
