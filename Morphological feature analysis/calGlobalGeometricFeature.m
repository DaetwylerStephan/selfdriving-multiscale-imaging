function [globalGeoFeature convexImage Image] = calGlobalGeometricFeature(image,varargin)
%calGeoFea calculates the geometric features for an image.
% It uses the regionprop3|regionprop for a 3D|2D image
% 
% INPUT       
% image    3D image (matrix)
% 
% OUTPUT      
% globalGeoFeature    structure array of global geometric features of an
%                     cell image
% convexImage         convex image of the cell (output of regionprop)
% Image               image of the cell without blancked frames
% 
% Hanieh Mazloom-Farsibaf, Gaudenz Danuser lab, 2020

%calculate the basic global features from regionprop3
if ~size(image,3) > 1 % 3D image
    error('Input should be 3D image')
else 
    s = regionprops3(image,"ConvexHull","Volume",'ConvexImage',...
        'ConvexVolume','Centroid',"BoundingBox",'Extent','Solidity','Image', ...
        'SurfaceArea','SubarrayIdx','EquivDiameter','PrincipalAxisLength');
    s(s.Volume==0,:)=[]; %to clean up the nonrelevent output of regionprop3
    globalGeoFeature.Volume=s.Volume; % in pixels^3 
    globalGeoFeature.SurfaceArea= s.SurfaceArea;
    globalGeoFeature.Solidity= s.Solidity; 
    globalGeoFeature.EquivDiameter=s.EquivDiameter;
    globalGeoFeature.CompactNess= s.SurfaceArea^1.5/s.Volume;
    globalGeoFeature.Sphericity= ((pi)^(1/3)*(6*s.Volume)^(2/3))/s.SurfaceArea;
    globalGeoFeature.Extent= s.Extent; %ratio of original volume to bounding box volume
    globalGeoFeature.Centroid=s.Centroid;
    [centerValue, centerLocation] = findInteriorMostPoint(image);
    globalGeoFeature.InteriorPoint=[centerValue, centerLocation];
    globalGeoFeature.AspectRatio=min(s.PrincipalAxisLength)/max(s.PrincipalAxisLength);

    %calculate roughness based on the created convexhull
    s_convHull=regionprops3(s.ConvexImage{1},'Volume','SurfaceArea');
    s_convHull(s_convHull.Volume==0,:)=[]; %to clean up the nonrelevent output of regionprop3
    globalGeoFeature.Roughness=s.SurfaceArea/s_convHull.SurfaceArea;

%calculate the longest length
    points3D=s.ConvexHull{end}; % vertex coordinate of a polygon
    distTemp=zeros(size(points3D,1));
    for ii=1:size(points3D,1)
        for jj=ii:size(points3D,1)
            distTemp(ii,jj)=sqrt((points3D(ii,1)-points3D(jj,1))^2 ...
                +(points3D(ii,2)-points3D(jj,2))^2+ ...
                (points3D(ii,3)-points3D(jj,3))^2);
        end
    end
    globalGeoFeature.NLongLength=max(distTemp(:))/s.Volume;
    globalGeoFeature.LongLength=max(distTemp(:));

    globalGeoFeature.NShortLength=min(distTemp(:))/s.Volume;
    globalGeoFeature.ShortLength=min(distTemp(:));
    
    %calculate the circumscribed sphericity, 
    [rows, cols, z] = findND(image); %Find non-zero elements in ND-arrays
    NonZeroMatrix = [rows, cols, z];
    [circumscribed_center, circumscribed_radius] = minboundsphere(NonZeroMatrix); %find the radius and center of the minimum bounding sphere using external function.
    %property of circumscribed Sphere
    CirumscribedSphere.center=circumscribed_center;
    CirumscribedSphere.radius=circumscribed_radius;
    CirumscribedSphere.volume=4/3 * pi * (circumscribed_radius);
    CirumscribedSphere.surfaceArea = 4 * pi * (circumscribed_radius ^2);
    globalGeoFeature.CirumscribedSphere=CirumscribedSphere;
    globalGeoFeature.CirmuscribedSurfaceRatio=globalGeoFeature.SurfaceArea/ ...
        CirumscribedSphere.surfaceArea
    
    %calculate the inscribed sphericity, (Roshan),
    [inscribed_radius, inscribed_center] = findInteriorMostPoint(image); %find the innermost point of the image; this will be
    %property of inscribed Sphere
    InscribedSphere.center=inscribed_center;
    InscribedSphere.radius=inscribed_radius;
    InscribedSphere.volume=4/3 * pi * (inscribed_radius);
    InscribedSphere.surfaceArea = 4 * pi * (inscribed_radius ^2);
    globalGeoFeature.InscribedSphere=InscribedSphere;
    
    %define the various sphericity
    VolumeSphericity=globalGeoFeature.Volume/CirumscribedSphere.volume;
    globalGeoFeature.VolumeSphericity=VolumeSphericity;
    
    RadiusSphericity=(globalGeoFeature.EquivDiameter/2)/CirumscribedSphere.radius;
    globalGeoFeature.RadiusSphericity=RadiusSphericity;
    
    RatioSphericity=InscribedSphere.radius/CirumscribedSphere.radius;
    globalGeoFeature.RatioSphericity=RatioSphericity;

end
%reduce the size of Image
Image=s.Image{end};
convexImage=s.ConvexImage{end};

end