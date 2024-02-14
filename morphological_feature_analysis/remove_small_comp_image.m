function [image] = remove_small_comp_image(image)
%remove_small_comp_image removes small components in a binary image, 2D or 3D

%find individual components
 CC = bwconncomp(image);
 numOfPixels = cellfun(@numel,CC.PixelIdxList);
 %find the largest components
 [~,indexOfMax] = max(numOfPixels);
    
 %find the index that have smaller components
 Ind = [1:CC.NumObjects]';
 %remove the largest components
 Ind(indexOfMax) =[];
 %find all the pixelIndex with smaller components
 PixelZeros = cat(1,CC.PixelIdxList{Ind}); 
 
 %make everywhere zero except the largest components 
 image(PixelZeros) = 0;
 