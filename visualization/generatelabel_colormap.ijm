
for (i = 0; i < 100; i++) {

close("*");
parentpath = "Z:/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013_stitched/fish3_segmented/1_CH488_000000";

timepoint =  "t" + IJ.pad(i, 5);
currenttimepointpath = parentpath + File.separator + timepoint;
open(currenttimepointpath + File.separator + "1_CH488_000000sg.tif");
firstimage = getTitle();

//set lut and divide
run("Rainbow Smooth");
setMinAndMax(0, 500);
run("RGB Color");
run("Split Channels");

//save new windows
selectWindow(firstimage + " (red)");
saveAs("Tiff", currenttimepointpath + File.separator + "1_CH488_000000sg_red.tif");
selectWindow(firstimage + " (green)");
saveAs("Tiff", currenttimepointpath  + File.separator + "1_CH488_000000sg_green.tif");
selectWindow(firstimage + " (blue)");
saveAs("Tiff", currenttimepointpath + File.separator + "1_CH488_000000sg_blue.tif");

close("*");
}