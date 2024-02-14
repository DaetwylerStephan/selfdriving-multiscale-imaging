function [clusterTend] = HopkinsStat(data,RandDist,nRand)
% HopkinsStat calculates the cluster tendency of a distribution, 0 is for
% highly clustered and 0.5 for random. ;
%
% Hopkins statistics is defined based on Cross, Jain(1982)
% https://doi.org/10.1016/B978-0-08-027618-2.50054-1
% check this reference for detail, (doi: 10.32614/RJ-2022-055)
%
% INPUT
% data          distribution (N x 2) or (N x 3) for 2D and 3D 
% nRand         choose a set of random number for calculating the Hopkins
% RandDist      random distribution in the allowed region
%
% OUTPUT
% clusterTend   cluster tendency, (for random distribution = 0.5)

%Hanieh Mazloom-Farsibaf, Danuser lab 2023

if ~exist('nRand','var')
    nRand = round(length(data))/10; % m = 10% n 
end 
if ~exist('d','var')
    d = size(data,2); % dimension of the data
end 

% create a random sample from the distribution
Ind = [1:size(data,1)]';
RandIndex =[randperm(length(Ind),nRand)]'; 
data_rand = data(RandIndex,:);

% create a random distribution in allowed region 
RandIndexdist = [randperm(length(RandDist),nRand)]';
RandDist_par = RandDist(RandIndexdist,:);

% calculate the nearest neighbor for random sample 
[Ind_NN  dist_NN] = knnsearch(data,data_rand,'K',2); % find the nearest neighbor in data for each element of data_rand
W = sum(dist_NN(:,2).^d); % should be the 2nd NN because it the first one is always zero! (itself)

% calculate the nearest neighbor for random distribution
[Ind_NN  dist_NN] = knnsearch(data,RandDist_par,'K',1);
U = sum(dist_NN(:,1).^d);

%cluster tendency
clusterTend = U/(U+W);






    