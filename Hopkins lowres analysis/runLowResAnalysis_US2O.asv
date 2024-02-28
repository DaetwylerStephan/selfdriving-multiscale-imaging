% script for running low resolution data
%% set directory
imageDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/lowRes/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013_stitched/fish3/segmentationWholeFish_StephanFijiCode/fish_volume'; 
saveDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/lowRes/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013_stitched/fish3/segmentationWholeFish_StephanFijiCode/Hopkins_d3_Sample30_N200'; 
xlsxDir = '/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/macrophage_control/20230602_Daetwyler_Xenograft/Experiment0013_stitched/fish3_segmented_xlsx/1_CH488_000000';

% 
% imageDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/lowRes/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001_stitched/fish2/segmentationWholeFish_StephanFijiCode/fish_volume';
% saveDirectory = '/archive/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/lowRes/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001_stitched/fish2/segmentationWholeFish_StephanFijiCode/Hopkins_d3_Sample30_N200';
% % centroid of cell in and excel file - cells have been segmented
% xlsxDir = '/archive/bioinformatics/Danuser_lab/Fiolka/LabMembers/Stephan/multiscale_data/xenograft_experiments/U2OS_WT/20220729_Daetwyler_U2OS/Experiment0001_stitched/fish2_segmented_xlsx/1_CH488_000000';
saveDirConv = [saveDirectory filesep 'fish_volume_convHull'];
saveDirSingleComp = [saveDirectory filesep 'fish_volume_singleComponent'];


if ~isdir(saveDirectory) mkdir(saveDirectory); end
if ~isdir(saveDirConv) mkdir(saveDirConv); end
if ~isdir(saveDirSingleComp) mkdir(saveDirSingleComp); end

%%

for iTime = 0:97
    
    % run for one timepoint
    iTime
    tic;
    s = sprintf('%02d',iTime);
    % loop for time points
    filename = ['fishvolume_t000' s '.tif'];

    % load the image3D
    image3D = load3DImage(imageDirectory, filename);
    %remove small components from image
    [image3D] = remove_small_comp_image(image3D);
    %save the convex hull image
    savename = [erase(filename, '.tif') '_SingleComp.tif'];
    save3DImage(image3D,fullfile(saveDirSingleComp,savename));
    Image3dMIP = sum(image3D,3);
    Image3dMIP(Image3dMIP~=0) = 1;
    savename = [erase(filename, '.tif') '_SingleCompMIP.tif'];
    imwrite(Image3dMIP,fullfile(saveDirSingleComp,savename));

    % phase 2: create the convexhull image
    image3D_ConvHull = create_convexHull3D(image3D);
    savename = [erase(filename, 'tif') '_bwconvhull.tif'];
    save3DImage(image3D_ConvHull,fullfile(saveDirConv,savename));
    Image3dMIP = sum(image3D_ConvHull,3);
    Image3dMIP(Image3dMIP~=0) = 1;
    savename = [erase(filename, '.tif') '_bwconvhullMIP.tif'];
    imwrite(Image3dMIP,fullfile(saveDirConv,savename));

    % flag for using the convexhull image
    params.useConvHulfile:///project/bioinformatics/Danuser_lab/zebrafish/analysis/Hanieh/Stephan/codes/lowResAnalysis/runLowResAnalysis_RandomDist.m
limage = 0;
    if params.useConvHullimage
        image = image3D_ConvHull;
    else
        image = image3D;
    end

    % phase 3,4,5:
    %load the xlx file
    ScaleFactor = 9.2; % downscale for segmentation was done!
    params.unit =  ''; % params.unit =  'micron'

    % for one time point
    timename = ['t000' s '.xlsx'];
    % sheet='Sarcoma+FibroblastNew55';
    sheet='Sheet1';
    [NumData]=xlsread(fullfile(xlsxDir, timename));

    % extract coordinates of each cell in 3D
    X= NumData(:,25); % in pixel
    Y= NumData(:,27);
    Z= NumData(:,29);

    % scale it properly
    X= NumData(:,25)*ScaleFactor; % in pixel
    Y= NumData(:,27)*ScaleFactor;
    Z= NumData(:,29);

    % convert it to the physical unit
    if strcmp(params.unit, 'micron')
        X= NumData(:,25)*pixelSize; % in um
        Y= NumData(:,27)*pixelSize;
        Z= NumData(:,29)*pixelSizeZ;
    end
    % coordinate of data points
    dataXYZ = [X Y Z];
    ParNum = size(X,1);
    NTrial = 200;
    %RandNum = ParNum*NTrial; % create more random position in allowed region then run the for loop for it later
    % generate the random point
    %run the cluster tendency
    p_sampling = 0.3;   % between 0 to 1
    nRand = round(p_sampling*ParNum);
    [RandIndex RandCoor_Total] = genRandomPixel3Dimage(image,nRand*NTrial);
    for iTrial= 1: NTrial
        startInd = (iTrial-1)*nRand+1;
        endInd = iTrial*(nRand);
        RandCoor = RandCoor_Total(startInd:endInd,:);
        [clusterTend(iTrial,1)] = HopkinsStat(dataXYZ,RandCoor,nRand);
    end
    savename = [erase(filename, '.tif') '_HopkinsResult.mat'];
    save(fullfile(saveDirectory, savename),'clusterTend','RandCoor_Total');

    % flag for using the convexhull image
    useConvHullimage = 1;
    if useConvHullimage
        image = image3D_ConvHull;
    else
        image = image3D;
    end
    % generate the random point

    %run the cluster tendency
    %      p_sampling = 0.3;   % between 0 to 1
    %     nRand = p_sampling*RandNum;
    nRand = round(p_sampling*ParNum);
    [RandIndex RandCoor_Total] = genRandomPixel3Dimage(image,nRand*NTrial);
    for iTrial= 1: NTrial
        startInd = (iTrial-1)*nRand+1;
        endInd = iTrial*(nRand);
        RandCoor = RandCoor_Total(startInd:endInd,:);
        [clusterTend(iTrial,1)] = HopkinsStat(dataXYZ,RandCoor,nRand);
    end
    
    savename = [erase(filename, '.tif') '_HopkinsResult.mat'];
    save(fullfile(saveDirConv, savename),'clusterTend','RandCoor_Total')
    t = toc
end