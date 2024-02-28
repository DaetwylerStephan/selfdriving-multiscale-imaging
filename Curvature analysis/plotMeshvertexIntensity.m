function plotMeshvertexIntensity(surface, data)
    if (nargin == 1)
        data = surface.vertices(:,1).*surface.vertices(:,2).*surface.vertices(:,3);
    end
    
    trimesh(surface.faces, surface.vertices(:,2), surface.vertices(:,3), ...
        surface.vertices(:,1), data, ...
        'EdgeColor', 'interp', 'FaceColor', 'interp');
    view([-221 24]);
    axis equal;
    axis off;
    
emax = max(data); % compute the minumum and maximum values for a consistent color map
emin = min(data);

caxis manual % use a consistent color map
% caxis([min(0.5,emin) max(1.5,emax)]);
caxis([emin emax]);

lighting flat;
shading flat;

camlight('headlight'); % create light in 3D scene

cb = colorbar;
set(cb,'position',[.92 .22 .01 .7]);
end