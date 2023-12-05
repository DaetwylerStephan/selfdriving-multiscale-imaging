# self-driving, multi-scale imaging
This repository contains the tools to process and analyse the multiscale data.


## Visualization
1. Generate the colormap as desired in Fiji (https://fiji.sc) using ImageJ's macro language (generatelabel_colormap.ijm) and save images to the folders containing the label image. <br> Here, we use the rainbow smooth lut (https://imagej.nih.gov/ij/images/luts-mbf/)
2. Use Napari to make the visualization of the 3D data over time. To visualize the low resolution data, use visualize_stitched_lowres.py
