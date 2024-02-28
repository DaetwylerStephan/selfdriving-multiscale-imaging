function [image3DStructure] = cropLabeledImage3D(image3D,p)
%cropLabeledImage3D crops individual cell in a multicell binary image
% INPUT 
% image3D   3Dimage or 3D matrix
% p         parameters including pixelEdge(pixel to define the edge)
%           rmEdgeCell (flag to remove the edge cell)
%           rmSmallComp (remove small componnets for a single cell image)
% OUTPUT
% image3DStructure  structure array if single cell image, includes two
%                   fields:image and Coordinate of cropped image in original image

% Hanieh Mazloom-Farsibaf , Danuser lab 2023


%remove cells touching the edge
pixelEdge = p.pixelEdge;
% find the number of labeled cells
Num_cell = unique(double(image3D(:)));

% recognize the cell close to the edge and remove them
if p.rmEdgeCell
    ImageSize = size(image3D);
    
    % find the edge pixed in x,y,z
    edgePixelsX = image3D([1:pixelEdge, ImageSize(1)-pixelEdge+1:ImageSize(1)],:,:);
    edgePixelsY = image3D(:,[1:pixelEdge, ImageSize(2)-pixelEdge+1:ImageSize(2)],:);
    edgePixelsZ = image3D(:,:,[1:pixelEdge, ImageSize(3)-pixelEdge+1:ImageSize(3)]);
    
    edgePixels = cat(1,edgePixelsX(:),edgePixelsY(:),edgePixelsZ(:));
    edgeLabel = unique(edgePixels);
    
    % remove the cell edge from the analysis
    Num_cell = Num_cell (~ismember(Num_cell,edgeLabel));
end
% crop the original image
for iCell = 1 : length(Num_cell)
    imagetmp = image3D;
    
    % create a mask for the labeled cell
    imagetmp (imagetmp ~= Num_cell(iCell)) = 0;
    imagetmp (imagetmp ~= 0) = 1;
    
    
    %remove small components of the image
    if p.rmSmallComp
        [imagetmp] = remove_small_comp_image(imagetmp);
    end
    %     % use the regionprop3 to crop the image automatically
    %     s = regionprops3 (imagetmp, 'Image');
    %     Image = s.Image{1};
    % instead find the coordinate - doesn't work
    Ind = find(imagetmp);
    [x y z] = ind2sub(ImageSize,Ind);
    Image = imagetmp([min(x): max(x)],[min(y):max(y)],[min(z):max(z)]);
    
    %     % add 1 pixel for each border this cause the problem!
    %     Image = addBlackBorder(Image,1);
    %
    % save in a structure array
    image3DStructure(iCell).Image = Image; %
    image3DStructure(iCell).Coor = [x y z]; % save the cell coordinate
    
    
end

