function image3DStructure=runCropLabeledImage3D(fileDir,filename,saveDir,p)

% load the image3D
image3D = load3DImage(fileDir, filename);

%crop the cells
[image3DStructure] = cropLabeledImage3D(image3D,p);

%save as tif file in the fileDir 
saveDirectory = [saveDir filesep 'CroppedCells' filesep]; 
if ~isdir(saveDirectory) mkdir(saveDirectory); end 
for iCell=1: size(image3DStructure,2)
    Image = image3DStructure(iCell).Image; 
    foldername = [saveDirectory filesep 'Cell' num2str(iCell) filesep];
    mkdir(foldername);
    filename = ['segmentedImage.tif'];
    save3DImage(Image, [foldername, filename])
end 

end

