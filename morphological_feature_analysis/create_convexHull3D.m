function image_ConvHull = create_convexHull3D(image)
% create_convexHull3D creates a convex hull of 3D image using 2D stack

%initialize the image 
image_ConvHull= zeros(size(image));

for iStack = 1:size(image,3)
    I = image(:,:,iStack);
    image_ConvHull(:,:,iStack) = bwconvhull(I);
end

