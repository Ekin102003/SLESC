function [Y] = SLESC(X,ind_folds,numClust,lambda1,lambda2,lambda3)

opts.record = 0;
opts.mxitr  = 30;%1000
opts.xtol = 1e-5;
opts.gtol = 1e-5;
opts.ftol = 1e-8;

    
max_iter = 100;
miu = 0.1;
rho = 1.1;

num_view = length(X);
for iv = 1:num_view
    X1 = X{iv}';
    X1 = double(X1);
    X1 = NormalizeFea(X1,1);
    ind_0 = find(ind_folds(:,iv) == 0);
    X1(ind_0,:) = [];       
    Y{iv} = X1';            
    W1 = eye(size(ind_folds,1));
    W1(ind_0,:) = [];
    G{iv} = W1;                                               
end
clear X X1 W1
X = Y;
clear Y      


nn =9;

    
for iv = 1:num_view
    options = [];
    options.NeighborMode = 'KNN';
    options.k = nn;
    options.WeightMode = 'Binary';
    Z1 = constructW(X{iv}',options);
    Z_ini{iv} = full(Z1);
    clear Z1;
end

for iv = 1:num_view
    invXX{iv} = inv(X{iv}'*X{iv}+2*eye(size(X{iv},2)));
end
F_ini = solveF(Z_ini,G,numClust);



num_view = length(X);
v = num_view;
[n, k] = size(F_ini{1});
Z = Z_ini;
S = Z;
W = Z;
F = F_ini;
Y = zeros(n, k);
Rs = cell(v, 1);
for idx = 1:v; Rs{idx} = eye(k); end
p = ones(v, 1);

for i = 1:num_view
    C1{i} = zeros(size(X{i}));
    C2{i} = zeros(size(Z{i}));
    C3{i} = zeros(size(Z{i}));
end

for iter = 1:max_iter
    %XXv = 0;
    % ------------ Y -------------- %
    sum_pFR = zeros(size(Y));
    for idx = 1:v
    sum_pFR = sum_pFR + F{idx}*Rs{idx}/p(idx);
    end
    [~, y_idx] = max(sum_pFR, [], 2);
    Y = full(sparse(1:n, y_idx, ones(n, 1), n, k));
      
    for i = 1:num_view
        % == update R_i ==
        [tmp_u, ~, tmp_v] = svd(Y'*F{i});
        Rs{i} = tmp_v * tmp_u';
    % == update p_i ==
        p(i) = norm(Y - F{i}*Rs{i}, 'fro'); 
        % --------------1 Z{i} ------------ % 
        G1 = X{i}+C1{i}/miu;
        G2 = S{i} - C2{i}/miu;
        G3 = W{i} - C3{i}/miu;
        Z{i} = invXX{i}*(X{i}'*G1+G2+G3);
        clear G1 G2 G3
        % ------------2 W{i} -------------- %
        P = G{i}*F{i};
        Q = L2_distance_1(P',P');
        M = Z{i}+C3{i}/miu;
        linshi_W = M-0.5*lambda1*Q/miu;
        linshi_W = linshi_W-diag(diag(linshi_W));
        for ic = 1:size(Z{i},2)
            ind = 1:size(Z{i},2);
            ind(ic) = [];
            W{i}(ic,ind) = EProjSimplex_new(linshi_W(ic,ind));
        end
        clear linshi_W P Q M ind ic
        % ----------------3 S{i} ---------------- %
        temp = Z{i}+C2{i}/miu;
        [UH,sigmaH,VH] = svd(temp,'econ');
        SU = sigmaH;
        canshu = 1;
        mu = miu;
        for time=1:size(sigmaH,1)
         if sum(diag(sigmaH(1:time,1:time)))>canshu/mu
            if time<size(sigmaH,1)
                if ((sum(diag(sigmaH(1:time,1:time)))-canshu/mu)/time)>=sigmaH(time+1,time+1)
                    tsigma=(sum(diag(sigmaH(1:time,1:time)))-canshu/mu)/time;
                    break
                end
            else
                tsigma=(sum(diag(sigmaH(1:time,1:time)))-canshu/mu)/time;
            end
         else
            if time==size(sigmaH,1)
                tsigma=0;
            end
         end
        end
    
       for cba=1:time
        sigmaH(cba,cba)=tsigma;
       end
       S{i} = UH*sigmaH*VH';
       clear temp AU VU SVP
        % -------------- 4 F{i} --------- %
        WW = (abs(W{i})+abs(W{i}'))*0.5;
        LL = diag(sum(WW))-WW;
        U = zeros(size(F{i},1));
        for iii = 1: num_view
           if iii == i
               continue
           end
           U = U + F{iii}*F{iii}';
        end
        U = max(U,U');
        M = lambda1*G{i}'*LL*G{i} - lambda3*U;
        M(isnan(M))=0;
        M(isinf(M))=1e5;
        F{i} = solveFO(F{i},@fun1,opts,lambda2/p(i),Y,Rs{i},M);
        % ----------- C1 C2 C3 --------- %
        leq1 = X{i}-X{i}*Z{i};
        leq2 = Z{i}-S{i};
        leq3 = Z{i}-W{i};
        C1{i} = C1{i} + miu*leq1;
        C2{i} = C2{i} + miu*leq2;
        C3{i} = C3{i} + miu*leq3;
        % ---------- obj --------- %
        %linshi_obj1 = linshi_obj1+sum(abs(SU))+lambda1*trace(F{i}'*G{i}'*LL*G{i}*F{i});
        %linshi_obj2 = linshi_obj2+norm(leq1,'fro')^2+norm(leq2,'fro')^2+norm(leq3,'fro')^2;
        %XXv = XXv + norm(X{i},'fro')^2;
    end
    % ----------------- obj ----------- %
    %obj(iter) = (linshi_obj1+linshi_obj2+lambda3*(num_view*num_clust-trace(U'*FFF*U)))/XXv;
    %clear FFF
    % ---------- miu ------------- %
    miu = min(rho*miu,1e8);
    %if iter > 2 && abs(obj(iter)-obj(iter-1))<1e-7
    %    iter
    %    break;
    %end
end

end


function [F,G]=fun1(P,alpha,Y,Q,L)
    G=2*L*P-2*alpha*Y*Q';
    F=trace(P'*L*P)+alpha*(norm(Y-P*Q,'fro'))^2;
end