function saveDAEfile(image3D, mesh, vertexColorsIndex, cmap, climits, savePath)

% saveDAEfile - saves a colored mesh as a collada dae file

% generate and invert the normals
mesh.normals = isonormals(image3D, mesh.vertices, 'negate');

% truncate the colormap
vertexColorsIndex(vertexColorsIndex < climits(1)) = climits(1);
vertexColorsIndex(vertexColorsIndex > climits(2)) = climits(2);

% generate the surface colors
minColor = min(vertexColorsIndex); maxColor = max(vertexColorsIndex);
if numel(cmap) > 3
    vertexColorsRGB = cmap(floor((length(cmap)-1)*((vertexColorsIndex-minColor)/(maxColor-minColor)))+1,:);
else
    vertexColorsRGB = repmat(cmap,length(mesh.vertices),1);
end
vertexColorsRGBA = [vertexColorsRGB, ones(length(vertexColorsRGB),1)];

% find the contents of the dae file
daeContents = makeDAEfile(mesh.vertices, mesh.faces, mesh.normals, vertexColorsRGBA);

% write the text file
fid = fopen(savePath, 'w');
fprintf(fid, daeContents);
fclose(fid);