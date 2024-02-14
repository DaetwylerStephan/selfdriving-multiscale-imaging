function [StatVec StatVec_Norm]=createStatMatrix(GeoFeaturesTable,FeatType,ExpCond)
% createStatMatrix extracts desired features from GeometryFeature Table for
% stattistical tests


%extract the experimental conditions;
if ~exist('ExpCond','var')
    ExpCond=unique(GeoFeaturesTable.ExpCondition);
end 

%check which FeatType is requested from GeoFeaturesTable
FieldName=fieldnames(GeoFeaturesTable);

%intialize the Stat Vec including features and cell ID and expCondition
StatVec=nan(size(GeoFeaturesTable,1),size(FeatType,2)+2);

for ii=1:length(FeatType)
    IndFeature{ii}=find(strcmp(FieldName,FeatType{ii}));
    nameFeature{ii}=FieldName(IndFeature{ii});
    StatVec(:,ii)=table2array(GeoFeaturesTable(:,IndFeature{ii}));
end

%label each ExpCond as a new condition for furthur analysis
for jj=1:length(ExpCond) % loop to count the experimental condition types
    ExpCondID=ExpCond{jj};
    Ind = find(strcmp(GeoFeaturesTable.ExpCondition,ExpCondID));
    StatVec(Ind,end) = jj;
end

% assign the cell ID to statvec for identifying each cellID
StatVec(:,end-1) = GeoFeaturesTable.cellID;

%normalized the statistical matrix, only include geometrical features
StatVec_Norm = zscore(StatVec(:,1:end-2)); % exclude the expCondition, and CellID from StatVec_Norm. 
