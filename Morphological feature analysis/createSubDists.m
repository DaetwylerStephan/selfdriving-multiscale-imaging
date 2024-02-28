function [dist1 dist2]=createSubDists(totalDist,cellLabel,ExpCond)
%createSubDists creates two distributions from the total dataset

%find the index of cells in statistical matrix
Ind1=find(ismember(cellLabel,ExpCond{1}));
Ind2=find(ismember(cellLabel,ExpCond{2}));

dist1=totalDist(Ind1,1:2);
dist2=totalDist(Ind2,1:2);

% %visualize the distributions
% figure
% plot(dist1(:,1),dist1(:,2),'.', 'MarkerSize',10)
% hold on 
% plot(dist2(:,1),dist2(:,2),'.', 'MarkerSize',10)
% hold off