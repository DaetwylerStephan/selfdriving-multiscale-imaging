function [pValue compDist comp2Dists]=bootstrapping2DistComp(dist1,dist2,Ntrials,metric,genSample)
%bootstrapping2DistComp calculates a pvalue for two distributions using the 
% permutation test (null hypothesis: distribution 1 and 2 are from the same
% distributions.
% INPUT
% dist1         distribution 1
% dist2         distribution 2
% Ntrials       number of creating permutation test to calculate the pvalue 
% metric        metric to calculate the pvalue, 'mean', 'median','TukeyMedian'
%               , 'ConvexHull'. (Default: 'TukeyMedian')
% genSample     sampling method, 'randomPick' (sampling with repeat),
%               'randomLabel' (keep total distribution and randomly label data),
%               (Dafault:'randomLabel')
% OUTPUT
% pValue        pvalue of two distribution 
% compDist      value of the metric for Ntrials permutation test
% comp2Dists    value of the metric for original distribution
% 
% Hanieh Mazloom-Farsibaf - Danuser lab 2021

%set default for this function
if ~exist('Ntrials','var')
    Ntrials=1000;
end 
if ~exist('metric','var')
    metric='TukeyMedian';
end
if ~exist('genSample','var')
    genSample='randomLabel';
end

compDist=nan(Ntrials,1);
% create a pool of data
dist_total=cat(1,dist1,dist2);

%
%visualize the distributions
figure
plot(dist1(:,1),dist1(:,2),'.', 'MarkerSize',10)
hold on
plot(dist2(:,1),dist2(:,2),'.', 'MarkerSize',10)
hold off
xlabel('PC 1')
ylabel('PC 2')
legend('dist1','dist2')
% calculate the metric of the original distibutios
switch metric
    case 'mean'
        comp2Dists=pdist2(mean(dist1),mean(dist2));
    case 'median'
        comp2Dists=pdist2(median(dist1),median(dist2));
        
    case 'TukeyMedian'
        tukmed1=halfmed(dist1(:,1),dist1(:,2));
        tukmed2=halfmed(dist2(:,1),dist2(:,2));
        comp2Dists=pdist2(tukmed1,tukmed2);
    case 'ConvexHull'
        %extract the convex hull
        k1=convhull(dist1);
        k2=convhull(dist2);
        %create polyshape for convex hull of dist 1 & 2
        PShape1=polyshape(dist1(k1,1),dist1(k1,2));
        PShape2=polyshape(dist2(k2,1),dist2(k2,2));
        %calculate the intersection of shape 1 & 2
        ShapeInter=intersect(PShape1,PShape2);
        %calculate the toral dist of current dist 1 & 2
        dist_totaltmp=cat(1,dist1,dist2);
        %calculate the convex hull of the total dist and create a
        %polygon
        k_total=convhull(dist_totaltmp);
        PShapeT=polyshape(dist_totaltmp(k_total,1),dist_totaltmp(k_total,2));
        %calculate the area of intersection and total dist
        AD=polyarea(ShapeInter.Vertices(:,1),ShapeInter.Vertices(:,2));
        AT=polyarea(PShapeT.Vertices(:,1),PShapeT.Vertices(:,2));
        
        %use polygon area as a comparison factor between two dist (intersection(1,2) & total)
        comp2Dists=AD/AT; % if same dist it is 1, if it is totaly different it is 0
        %limitation, it doesn't capture how far they are if there is no
        %intersection
end

% Do the boostrapping to find a distribution of the metric
for nn=1:Ntrials
    clear Ind1 Ind2 tukmed1 tukmed2
    switch genSample
        case 'randomPick'
            Ind1=randi([1 length(dist_total)],length(dist1),1);
            Ind2=randi([1 length(dist_total)],length(dist2),1);
            disttmp1=dist_total(Ind1,:);
            disttmp2=dist_total(Ind2,:);
        case 'randomLabel' %permutation test
            Ind1=randperm(length(dist_total),length(dist1));
            disttmp1=dist_total(Ind1,:);
            dist_totaltmp=dist_total;
            dist_totaltmp(Ind1,:)=nan;
            disttmp2=dist_totaltmp(~isnan(dist_totaltmp));
            disttmp2=reshape(disttmp2,[size(dist2,1),size(dist2,2)]);
            
    end
    switch metric
        case 'mean'
            compDist(nn)=pdist2(mean(disttmp1),mean(disttmp2));
            %          pValue=sum(compDist > comp2Dists)/size(compDist,1);
            
        case 'median'
            compDist(nn)=pdist2(median(disttmp1),median(disttmp2));
            
            %calculate the pvalue to compare dist1,dist2,
            %pvalue in this case is right tail
            % pValue=sum(compDist > comp2Dists)/size(compDist,1);
        case 'TukeyMedian'
            try
            tukmed1=halfmed(disttmp1(:,1),disttmp1(:,2));
            tukmed2=halfmed(disttmp2(:,1),disttmp2(:,2));
            compDist(nn)=pdist2(tukmed1,tukmed2);
            catch ME
                continue
            end 
        case 'ConvexHull'
            %extract the convex hull
            k1=convhull(disttmp1);
            k2=convhull(disttmp2);
            %create polyshape for convex hull of dist 1 & 2
            PShape1=polyshape(disttmp1(k1,1),disttmp1(k1,2));
            PShape2=polyshape(disttmp2(k2,1),disttmp2(k2,2));
            %calculate the intersection of shape 1 & 2
            ShapeInter=intersect(PShape1,PShape2);
            %calculate the toral dist of current dist 1 & 2
            dist_totaltmp=cat(1,disttmp1,disttmp2);
            %calculate the convex hull of the total dist and create a
            %polygon
            k_total=convhull(dist_totaltmp);
            PShapeT=polyshape(dist_totaltmp(k_total,1),dist_totaltmp(k_total,2));
            %calculate the area of intersection and total dist
            AD=polyarea(ShapeInter.Vertices(:,1),ShapeInter.Vertices(:,2));
            AT=polyarea(PShapeT.Vertices(:,1),PShapeT.Vertices(:,2));
            
            %use polygon area as a comparison factor between two dist (intersection(1,2) & total)
            compDist(nn)=AD/AT; % if same dist it is 1, if it is totaly different it is 0
            %limitation, it doesn't capture how far they are if there is no
            %intersection
            AD_t(nn)=AD; AT_t(nn)=AT;
            %calculate the pvalue to compare dist1,dist2 is left tail
            %             pValue=sum(compDist < comp2Dists)/size(compDist,1);
        case 'EMD'
            % define the weight for each significant. (significant is the
            % distribution for the EMD)
            
            %calculate the mode for each cluster(value)
            
            %calculate the fraction of each distibution within the cluster
            %(weight)
            
        otherwise
            
    end
    
    
end

%calculate pvalue

switch metric
    case 'mean'
        pValue=sum(compDist > comp2Dists)/size(compDist,1);
        
    case 'median'
        pValue=sum(compDist > comp2Dists)/size(compDist,1);
        
    case 'TukeyMedian'
        pValue=sum(compDist > comp2Dists)/size(compDist,1);
        
    case 'ConvexHull'
        pValue=sum(compDist < comp2Dists)/size(compDist,1);
    otherwise
end
end