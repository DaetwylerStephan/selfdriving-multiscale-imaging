function [centerValue, centerLocation] = findInteriorMostPoint(image3DBinary)

% findMostInteriorPoint - given a binary 3D image of a cell, find the interior point that is farthest from the cell edge 

% fill the binary image
image3DBinary = imfill(image3DBinary, 'holes');

% find the distance from the cell edge
imageDist = bwdist(~image3DBinary);

% find the value and location of the interior-most point 
% (not stable if there is more than one such point)
centerValue = max(imageDist(:));
locationIndex = find(imageDist==centerValue, 1);
[iLoc, jLoc, kLoc] = ind2sub(size(imageDist), locationIndex);
centerLocation = [iLoc, jLoc, kLoc];
