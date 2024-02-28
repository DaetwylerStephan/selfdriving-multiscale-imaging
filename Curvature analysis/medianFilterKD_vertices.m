function medianFiltered = medianFilterKD_vertices(surface, measure, radius)

% medianFilterKD - Median filter the mesh in real 3-D space


% % get the face center positions 
nVertex = size(surface.faces,1);
% vertex = surface.vertices;
% for f = 1:nFaces
%     faceCenters(f,:) = mean(surface.vertices(surface.faces(f,:),:),1);
% end

% find points within the averaging radius of each surface face
iClosest = KDTreeBallQuery(surface.vertices,surface.vertices,radius);

% median filter the data
medianFiltered = zeros(nVertex,1);
for j = 1:numel(iClosest)
    medianFiltered(j,1) = median(measure(iClosest{j}));
end