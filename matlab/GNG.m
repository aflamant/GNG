classdef GNG
    properties
        N
        MaxIt
        L
        epsilon_b
        epsilon_n
        alpha
        delta
        T
        PlotFlag
    end
    methods
        function obj = GNG(N, MaxIt, L, epsilon_b, epsilon_n, alpha, delta, T, PlotFlag)
            if nargin ~= 0
                obj.N = N;
                obj.MaxIt = MaxIt;
                obj.L = L;
                obj.epsilon_b = epsilon_b;
                obj.epsilon_n = epsilon_n;
                obj.alpha = alpha;
                obj.delta = delta;
                obj.T = T;
                obj.PlotFlag = PlotFlag;
            else
                obj.N = 50;
                obj.MaxIt = 20;
                obj.L = 50;
                obj.epsilon_b = 0.2;
                obj.epsilon_n = 0.005;
                obj.alpha = 0.5;
                obj.delta = 0.995;
                obj.T = 20;
                obj.PlotFlag = true;
            end
        end
        function net = fit_gas(obj, X)
            %% Load Data
    
            nData = size(X,1);
            nDim = size(X,2);

            X = X(randperm(nData), :);

            Xmin = min(X);
            Xmax = max(X);
            
            %% Initialization

            Ni = 2;

            w = zeros(Ni, nDim);
            for i = 1:Ni
                w(i,:) = unifrnd(Xmin, Xmax);
            end

            E = zeros(Ni,1);

            C = zeros(Ni, Ni);
            t = zeros(Ni, Ni);

            %% Loop

            nx = 0;

            for it = 1:obj.MaxIt
                for l = 1:nData
                    % Select Input
                    nx = nx + 1;
                    x = X(l,:);

                    % Competion and Ranking
                    d = pdist2(x, w);
                    [~, SortOrder] = sort(d);
                    s1 = SortOrder(1);
                    s2 = SortOrder(2);

                    % Aging
                    t(s1, :) = t(s1, :) + 1;
                    t(:, s1) = t(:, s1) + 1;

                    % Add Error
                    E(s1) = E(s1) + d(s1)^2;

                    % Adaptation
                    w(s1,:) = w(s1,:) + obj.epsilon_b*(x-w(s1,:));
                    Ns1 = find(C(s1,:)==1);
                    for j=Ns1
                        w(j,:) = w(j,:) + obj.epsilon_n*(x-w(j,:));
                    end

                    % Create Link
                    C(s1,s2) = 1;
                    C(s2,s1) = 1;
                    t(s1,s2) = 0;
                    t(s2,s1) = 0;

                    % Remove Old Links
                    C(t>obj.T) = 0;
                    nNeighbor = sum(C);
                    AloneNodes = (nNeighbor==0);
                    C(AloneNodes, :) = [];
                    C(:, AloneNodes) = [];
                    t(AloneNodes, :) = [];
                    t(:, AloneNodes) = [];
                    w(AloneNodes, :) = [];
                    E(AloneNodes) = [];

                    % Add New Nodes
                    if mod(nx, obj.L) == 0 && size(w,1) < obj.N
                        [~, q] = max(E);
                        [~, f] = max(C(:,q).*E);

                        r = size(w,1) + 1;
                        w(r,:) = (w(q,:) + w(f,:))/2;

                        C(q,f) = 0;
                        C(f,q) = 0;

                        C(q,r) = 1;
                        C(r,q) = 1;
                        C(r,f) = 1;
                        C(f,r) = 1;

                        t(r,:) = 0;
                        t(:, r) = 0;
                        E(q) = obj.alpha*E(q);
                        E(f) = obj.alpha*E(f);
                        E(r) = E(q);
                    end

                    % Decrease Errors
                    E = obj.delta*E;
                end

                % Plot Results
                if obj.PlotFlag
                    figure(1);
                    PlotResults(X, w, C);
                    pause(0.01);
                end
            end

            %% Export Results
            net.w = w;
            net.E = E;
            net.C = C;
            net.t = t;
        end
    end
end