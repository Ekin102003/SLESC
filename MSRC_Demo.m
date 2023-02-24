
clear;


Dataname = 'MSRC_Per0_3';
Datafold = 'MSRC_Folds';
load(Dataname);
load(Datafold);
ind_folds = folds{3};
truthF = truelabel{1};  
numClust = length(unique(truthF));
X = data;
lambda1  = 0.001;
lambda2  = 0.000001;
lambda3  = 0.01;

  


[preY] = SLESC(X,ind_folds,numClust,lambda1,lambda2,lambda3);
[~, preY] = max(preY, [], 2);
result = ClusteringMeasure(truthF, preY)    
