function image3D = load3DImage(inDirectory, name)

% load3DImage - load a 3D image using Matlab's built-in image reader
%
% INPUTS:
%
% inDirectory - the path to the image
%
% name - the name of the image

% try to find information about the image
try
    imageInfo = imfinfo(fullfile(inDirectory, name));
catch 
    disp([name ' is not an image and will not be analyzed.'])
    image3D = [];
    return
end

% find the image size
imageSize = [imageInfo(1).Height; imageInfo(1).Width; length(imageInfo)];

% initiate the image variable
image3D = zeros(imageSize(1), imageSize(2), imageSize(3));

% load each plane in Z
parfor z = 1:imageSize(3) 
    image3D(:,:,z) = im2double(imread(fullfile(inDirectory, name), 'Index', z, 'Info', imageInfo));
%     image3D(:,:,z) = imread(fullfile(inDirectory, name), 'Index', z, 'Info', imageInfo);

end
