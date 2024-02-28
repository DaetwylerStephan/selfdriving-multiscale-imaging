%% run a random dist inside whole fish and calculate the Hopkins 

% set the directory 
fileDir = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/lowRes/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013_stitched/fish3/segmentationWholeFish_StephanFijiCode/fish_volume_singleComponent'; 
saveDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/lowRes/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013_stitched/fish3/segmentationWholeFish_StephanFijiCode/Testcode/fish_volume_RandomDistHopkins_d3_p30_N350'; 
if ~isdir(saveDirectory) mkdir(saveDirectory); end 


% for each time point 

% load the whole volume segmented fish, it can be US20 or WT fish (so simliar)
 
RandDistNum = 400; % number od cell for a random distribution
NTrial = 350; % calculate the Hopkins for Ntrials
useConvHullimage = 0;

for iTime = 0:97
tic
    % run for one timepoint
    iTime
    s = sprintf('%02d',iTime);
    % loop for time points
    filename = ['fishvolume_t000' s '_SingleComp.tif'];
    
    % load the image3D
    image3D = load3DImage(fileDir, filename);
    
%     % phase 2: create the convexhull image
%     image3D_ConvHull = load3DImage(saveDirConv,savename);
%   
    % flag for using the convexhull image
    if useConvHullimage
        image = image3D_ConvHull;
    else
        image = image3D;
    end
    
   
    % for one time point
    timename = ['t000' s '.xlsx'];

    % generate the random distribution within this fish volume 
    [RandIndex dataXYZ ] = genRandomPixel3Dimage(image,RandDistNum);
    
    %run the cluster tendency
    p_sampling = 0.3;   % percent of sampling random for Hopkins stat between 0 to 1 
    nRand = round(p_sampling*RandDistNum);
   
    % this one is for Hopkins stat, 
    [RandIndex RandCoor_Total] = genRandomPixel3Dimage(image,nRand*NTrial);
    for iTrial= 1: NTrial
        startInd = (iTrial-1)*nRand+1;
        endInd = iTrial*(nRand);
        RandCoor = RandCoor_Total(startInd:endInd,:);
        [clusterTend(iTrial,1)] = HopkinsStat(dataXYZ,RandCoor,nRand);
    end
    toc
    savename = [erase(filename, '.tif') '_HopkinsResult.mat'];
    save(fullfile(saveDirectory, savename),'clusterTend','RandCoor_Total');

    savename = [erase(filename, '.tif') 'RandomDist.mat'];
    % image3DRand = single(image3DRand);
    save(fullfile(saveDirectory, savename),'dataXYZ')  
  t= toc 
end
