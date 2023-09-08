
//macro for generating the fish volume based on vascular data for analysis

//-------------------------------------------------
// parameters
//-------------------------------------------------
parentfolder = "/endosome/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001_stitched/fish1/"
imagename = "1_CH594_000000.tif"
parentfolder= "Z:/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001_stitched/fish1"
savefolder= "Z:/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001_stitched/fish_volume"
savefolder_max= "Z:/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001_stitched/fish_volume_max"
//-------------------------------------------------
// processing loop
//-------------------------------------------------

//make folder
File.makeDirectory(savefolder);
File.makeDirectory(savefolder_max);

for (i_time = 0; i_time < 7; i_time++) {
 	currentfilename = parentfolder + File.separator + "t" + IJ.pad(i_time, 5) + File.separator + imagename;
 	print(currentfilename);
 	
 	//open file
 	run("Bio-Formats Windowless Importer", "open="+currentfilename);
 	
 	//run Gamma correction
 	run("Gamma...", "value=0.90 stack");
 	//Gaussian Blur
 	run("Gaussian Blur 3D...", "x=10 y=10 z=3");
 	
 	//get 10% of highest values (percentage 90)
	percentage = 90; 
	nBins = 65536; 
	resetMinAndMax(); 
	stackHisto = newArray(65536);
	for ( j=1; j<=nSlices; j++ ) {
	    setSlice( j );
	    getHistogram(values, counts, 65536);
	    for ( i_stack=0; i_stack<65536; i_stack++ )
	       stackHisto[i_stack] += counts[i_stack];
	}	
	nPixels= getWidth() * getHeight() * nSlices;
	nBelowThreshold = nPixels * percentage / 100;
	
	sum = 0; 
	for (iter = 0; iter<stackHisto.length; iter++) { 
	  sum = sum + stackHisto[iter]; 
	  if (sum >= nBelowThreshold) { 
	    setThreshold(iter, 65535); 
	    print(iter + "-" + "65535: "  +sum/nPixels*100+"%"); 
	    iter = 99999999;//break 
	  } 
	} 
	print("threshold applied");
 	
 	//run("Threshold...");
 	run("Convert to Mask", "method=Default background=Dark");
 	//fill holes
 	run("Invert LUT");
 	run("Fill Holes", "stack");
 	//run second blurring
 	run("Gaussian Blur 3D...", "x=20 y=20 z=5");
 	setThreshold(5, 255);
 	//run("Threshold...");
 	run("Convert to Mask", "method=Default background=Dark");
 	
 	//fill holes
 	run("Invert LUT");
 	run("Fill Holes", "stack");
 	
 	saveAs("Tiff",savefolder + File.separator + "fishvolume_t" + IJ.pad(i_time,5) + ".tif");
 	
 	//make max projection for quick check
 	run("Z Project...", "projection=[Max Intensity]");
 	saveAs("Tiff",savefolder_max + File.separator + "fishvolume_t" + IJ.pad(i_time,5) + ".tif");
 	close("*");
}