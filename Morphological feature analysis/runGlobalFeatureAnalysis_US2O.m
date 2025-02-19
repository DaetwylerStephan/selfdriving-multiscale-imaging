% runGlobalFeatureAnalysis_US2O
% this script runs all functions for global geometric features for a series
% of images captured in zebrafish. 

imageDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/HighRes/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment00001_highres_manuallyCompiled2/high_stack_002';
saveDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/HighRes/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment00001_highres_manuallyCompiled2/high_stack_002/TestCodes_check';

timePoints = [0:49]; 
%% Step1: calculate the global geometrical features
filename = 'segmentedImage.tif'; % filename for cropped single cell

% loop across time series of images
for iTime = 1:length(timePoints)
    %set the current timepoint
    currTime = timePoints(iTime)
    s = sprintf('%02d',currTime);
    fileDirTime = [imageDirectory filesep 't000' s filesep 'CroppedCells'];
    saveDirTime = [saveDirectory filesep 't000' s filesep 'CroppedCells'];

    %check the number of cell in each timepoint folder
    dirInfo = dir(fileDirTime); 
    dirInfoName = extractfield(dirInfo,'name'); 
    fileList = dirInfoName(3:end)'; % exclude '.' and '..' from the list
    cellList = str2double(erase(fileList, 'Cell'));
    cellList = sort(cellList); 

    %loop across cell list for each timepoint
    for iCell=1:length(cellList)
        CellID = cellList(iCell)
        cellPath = [fileDirTime filesep 'Cell' num2str(CellID)]; 
        %check if cell exist in the folder then calculate the geometrical
        %features 
        if ~isfile(fullfile(cellPath,filename))
            warning('Tif file (image) of the cell is not in the path')
            break
        end 
        
        %load the segmented image
        image3D=load3DImage(cellPath,filename);
        
        %create GlobalMorphology folder for each cell
        saveCellPath=[saveDirTime filesep 'Cell' num2str(CellID) filesep 'GlobalMorphology'];
        if ~isfolder(saveCellPath)
            mkdir(saveCellPath);
        end

        %caculate the global feature for 3D image
        [globalGeoFeature convexImage Image] = calGlobalGeometricFeature(image3D);
        %save the global feature for each cell
        save(fullfile(saveCellPath,'globalGeoFeature.mat'),'globalGeoFeature');
        save(fullfile(saveCellPath,'convexImage.mat'),'convexImage');
        save(fullfile(saveCellPath,'Image.mat'),'Image');
    end
end

%% Step2: create a table of all cells in all timepoints for post analysis
%set the desired feature for the post-processing analysis
featType=[ {'Volume'} {'SurfaceArea'} ...
    {'Sphericity'} {'Solidity'} {'LongLength'} {'AspectRatio'}...
    {'Roughness'} {'Extent'} {'CirmuscribedSurfaceRatio'} ...
    {'VolumeSphericity'} {'RadiusSphericity'} {'RatioSphericity'}];

%initialize geometric feature table
GeoFeaturesTable =[];

%fill geometric feature table across time points
for iTime = 1:length(timePoints)
    currTime = timePoints(iTime);
    s = sprintf('%02d',currTime);
    saveDirTime = [saveDirectory filesep 't000' s filesep 'CroppedCells'];

    %check the number of cell in each timepoint folder
    dirInfo = dir(saveDirTime); 
    dirInfoName = extractfield(dirInfo,'name'); 
    fileList = dirInfoName(3:end)'; % exclude '.' and '..' from the list
    cellList = str2double(erase(fileList, 'Cell'));
    cellList = sort(cellList); 

    GeoFeaturesTabletmp=createGeoFeaturesTable(saveDirTime,cellList,featType);
    
    % add timepoint as an experimental condition
    GeoFeaturesTabletmp.ExpCondition(:) = {s}; 

    GeoFeaturesTable = [GeoFeaturesTable; GeoFeaturesTabletmp];
end 

saveDirTotal = [saveDirectory filesep 'Results'];
if ~isdir(saveDirTotal) mkdir(saveDirTotal); end
save(fullfile(saveDirTotal, 'GeoFeatureTable.mat'),'GeoFeaturesTable'); 


%% Step3: Statistical analysis for evaluation - pca 

%create a matrix of desired features (column) for a list of cells (row)
[statMat statMat_Norm]=createStatMatrix(GeoFeaturesTable,featType);

% PCA
plotFlag=1;
[coeff,score,eigenValues,tsquared,Contribution,mu]=calPCA_globalGeometry(statMat_Norm,plotFlag,featType);

save(fullfile(saveDirTotal,'pcaAllResults.mat'),'statMat','statMat_Norm','score',...
    'tsquared','coeff','mu','eigenValues','Contribution')

%% Step4: similarity of cell shape for entire data 

% pooling data for one-hour imaging
dt = 3; % == 1 hour for time resolution

%create a new time vector for combining dt timepoint together
Time = statMat(:,end); % last column in statMat is a label list represnting time
T_max = max(Time);
Time_New = nan(size(Time));
timecounter = 0; 
for iT = 1:dt:T_max % in 
    timecounter = timecounter +1;
    TimeLabel_tmp = [iT: iT+dt-1];
    Ind = find(ismember(Time,TimeLabel_tmp));
    Time_New (Ind) = timecounter;
end

%relabel the time point in statMat
statMat(:,end) = Time_New;

% calculate similarity matrix 
N_dim = 2; % calculate the pvalue for N_dim of pc space
endTime= max(statMat(:,end)); % 
startTime = min(statMat(:,end));

%boosttrapping parameters for pvalue
Ntrials=300;  % number of iteration 
metric='TukeyMedian';  % other matric: ConvexHull , mean ,TukeyMedian
genSample='randomLabel'; %method to sample the point for boostrapping

%initialize the pvalueMatrix
Pvalue_matrix = nan(max(Time_New));

% calculate the similarity matrix
for iTime = startTime:endTime
    time1 = iTime;
    Ind = find(statMat(:,end) == time1);

    %define dist1 for arbitary time point
    dist1 = score(Ind,1:N_dim);

    %     define dist2 for another time point
    for iTime2 = startTime:endTime
        time2 = iTime2;
        if time2 ~= time1
            Ind = find(Time_New == time2);
            %         define dist2 for arbitary time point
            dist2 = score(Ind,1:N_dim);
   
            [pValue compDist comp2Dists] =bootstrapping2DistComp(dist1,dist2,Ntrials,metric,genSample);
            %iTime starts from 0
            Pvalue_matrix(iTime+1,iTime2+1) = pValue;
        end
    end

end


save(fullfile(saveDirTotal,'pValueMatrix.mat'),'Pvalue_matrix')
