function [RandIndex RandCoor ] = genRandomPixel3Dimage(image3D,RandNum)
%generates random pixels from a segmented binary image

% Hanieh Mazloom-Farsibaf, Danuser lab 2023

%find the image size
SZ = size(image3D);

%find the index which has value one in the binary image3D
Ind_1 = find(image3D);

% create random number from nonzero value of image3D
% find random num of the index of the Ind_1 
RandIndex = randperm(length(Ind_1),RandNum); 
%find the index of the non-zero region
RandIndex = Ind_1(RandIndex); % random index of non-zero region

% check the pixel value is one 
RandPixels = image3D(RandIndex); % check if all the random pixel is non-zero
[Y X Z] = ind2sub(SZ,RandIndex); % because it is permute the x and y for the image
RandCoor = cat(2,X,Y,Z);

% %check the random position 
% image3DRand = zeros(size(image3D));
% image3DRand(RandIndex) = 10; 
% dipshow(image3DRand)