function save3DImage(image3D, imagePath)

% saveImage3D - saves a 3D image as a single tif (assumes that the image is already of the correct class)
%
% INPUTS:
%
% image3D - the image, which should be of the correct class already
%
% imagePath - the path and name of the image

% check inputs
assert(isnumeric(image3D), 'image3D must be a numerical matrix');
assert(ischar(imagePath), 'imagePath should be a path to an image');

% saves the first slice and overwrites any existing image of the same name
imwrite(squeeze(image3D(:,:,1)), imagePath, 'Compression', 'none')

% saves subsequent slices
imageSize = size(image3D);
for z=2:imageSize(3)
    imwrite(squeeze(image3D(:,:,z)), imagePath, 'Compression', 'none', 'WriteMode', 'append')

end