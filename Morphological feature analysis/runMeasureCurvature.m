% run_curvature
% This script calculates the mean curvature for a 3D images of single cell
% in zebrafish

%% step1: set directory
imageDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/ForPaper/cells_to_calculate/TestCode';
saveDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/ForPaper/cells_to_calculate/TestCode/Curvature';
filename = 't0001_high001_Cell_0001.tif'; 

if ~isdir(saveDirectory) mkdir(saveDirectory); end 


% step2: create a triangle mesh from a 3D image (single cell)
image3D = load3DImage(imageDirectory,filename);
surface = isosurface(image3D);


% step3: smooth the triangle mesh
numIterations = 100; 
smoothMethod = 'Taubin'; %'Average', 'Laplacian','Taubin'
%redefine the mesh for having a proper input for smoothSurfaceMesh 
mesh = surfaceMesh(surface.vertices,surface.faces);
surfaceMeshOut = smoothSurfaceMesh(mesh,numIterations, "Method",smoothMethod);
surfaceSmooth.vertices = surfaceMeshOut.Vertices;
surfaceSmooth.faces = surfaceMeshOut.Faces;


% step4: measure curvature using pricipal curvature
getderivatives = 0 
[PrincipalCurvatures,PrincipalDir1,PrincipalDir2,FaceCMatrix,VertexCMatrix,Cmagnitude]= GetCurvatures( surfaceSmooth ,getderivatives);
GausianCurvatureUnsmoothed_vertex=PrincipalCurvatures(1,:).*PrincipalCurvatures(2,:);
meanCurvatureUnsmoothed_vertex=(PrincipalCurvatures(1,:)+PrincipalCurvatures(2,:))/2;

% convert curvature to physical value
pixelSize = 0.4; % projected pixelsize on camera in um
meanCurvatureUnsmoothed_vertex = meanCurvatureUnsmoothed_vertex/pixelSize; 

% remove high and low curvature for visualization purpose
meanCurvatureUnsmoothed_vertex (isnan(meanCurvatureUnsmoothed_vertex)) = nanmean(meanCurvatureUnsmoothed_vertex);
meanCurvatureUnsmoothed_vertex(meanCurvatureUnsmoothed_vertex<prctile(meanCurvatureUnsmoothed_vertex,1)) = prctile(meanCurvatureUnsmoothed_vertex,1); 
meanCurvatureUnsmoothed_vertex(meanCurvatureUnsmoothed_vertex< -1) = -1;
meanCurvatureUnsmoothed_vertex(meanCurvatureUnsmoothed_vertex > 1) = 1;
% figure; 
% plotMeshvertexIntensity(surfaceSmooth,meanCurvatureUnsmoothed_vertex)
 
% step5: visualization, creating dae files for rendering in ChimeraX 
cmap = flipud(makeColormap('div_rwb', 1024));
climits = [-1 1];
daeSavePath=[saveDirectory filesep erase(filename,'.tif') 'srfSmooth' num2str(numIterations) '' num2str(climits) ...
     'UnsmoothSaturate.dae'];
saveDAEfile(image3D, surfaceSmooth, meanCurvatureUnsmoothed_vertex, cmap, climits, daeSavePath);

