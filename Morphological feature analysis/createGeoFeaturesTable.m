function GeoFeaturesTable=createGeoFeaturesTable(fileDirectory,cellID,featType)
%createGeoFeaturesTable creates the GeoFeaturesTable for statistical analysis
% 
% INPUT: 
% fileDirectory     file Directory of cells
% cellID,           a list of files (cell)
% featType          a list of desired features
% 
% OUTUT: 
% GeoFeaturesTable, a table of global features for a set of images
%
% Hanieh Mazloom-Farsibaf, 2020

%create an empty table for geometrical features based on the input features
GeoFeaturesTable = table(cellID);
emptyNan = nan(size(cellID));
for iFeature = 1: length(featType)
    featName = featType{iFeature};
    GeoFeaturesTable.(featName) = emptyNan;
end 

%fill the table for all sets of images 
for iCell=1:length(cellID) % loop for each file set 
  ID= cellID(iCell);
  cellPath = [fileDirectory filesep 'Cell' num2str(ID)];
       if ~isdir([cellPath filesep 'GlobalMorphology'])
           break
       end
       load(fullfile([cellPath filesep 'GlobalMorphology'],'globalGeoFeature'));
       
       %loop across the feature 
       for iFeat = 1: length(featType)
           featName = featType{iFeat};
           featValue = getfield(globalGeoFeature,featName);
           GeoFeaturesTable.(featName)(iCell) = featValue;
       end
end 