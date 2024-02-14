function [coeff,score,eigenValues,tsquared,Contribution,mu]=calPCA_globalGeometry(statMat_Norm,plotFlag,featType)
%calPCA_globalGeometry calculates the pca of global geometrical features

if ~exist('plotFlag','var')
plotFlag=1;
end

[coeff,score,eigenValues,tsquared,Contribution,mu] = pca(statMat_Norm);

if plotFlag
%plot the biplot
figure;
biplot(coeff(:,1:2),'Scores',score(:,1:2),'VarLabels',featType);

%plot the contribution histogram
    figure;
    plot(score(:,1),score(:,2),'.')
title('Global gemotry feature in principal component space')
xlabel('Component 1')
ylabel('Component 2')

%plot the contribution
name = categorical({'PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7',...
   'PCA8','PCA9','PCA10','PCA11','PCA12'});
name= reordercats(name,{'PCA1','PCA2','PCA3','PCA4','PCA5','PCA6','PCA7',...
   'PCA8','PCA9','PCA10','PCA11','PCA12'});
figure;
bar(name,Contribution)
ylabel('Contribution (%)')
title('histogram of PCAs contribution')

end 
