function [ yPredTest, yPredTrain, lambda, b, q, H, trainAcc, testAcc, nsv, time ] = MCM(xTrain, yTrain, xTest, yTest, Kernel_Flag, C, beta)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% MCM Kernel version 1.0 
% [Jayadeva. "Learning a hyperplane classifier by minimizing an exact bound on the VC dimensioni." NEUROCOMPUTING 149 (2015): 683-689.]
% Dr. Jayadeva, EE Department, IIT Delhi
%
% This code performs binary classification using MCM, on the given data.
%
% The input variables are:
% For training data :
% xTrain : data matrix(Ntrain X D) -> rows are data points, columns are features
% yTrain : Column label vector(Ntrain X 1) -> labels must be +1 and -1 (
% For testing data : 
% xTest : data matrix(Ntest X D) -> rows are data points, columns are features
% yTest : Column label vector(Ntest X 1) -> labels must be +1 and -1
% Kernel_Flag : 0 -> linear kernel, 1 -> rbf kernel
% C : specifies the value of constant C in the MCM formulation
% beta : specifies the width of the rbf kernel : the higher the value, the
% flatter(less flexible) the kernel
%
% The output variables are: 
% yPredTest: Predicted labels for test data
% yPredTrain: Predicted labels for train data
% lambda: lambda values corresponding to each data point in the training
% data
% b: bias constant for trained model
% q: slack variable values for trained model
% H: value of H, in the trained model
% trainAcc: in-sample prediction accuracy
% testAcc: out-of-sample prediction accuracy
% nsv: Number of support vectors in the training data
% time: Time taken by MCM
% 
% If you have any questions to ask or bugs to report, please email
% jayadeva@ee.iitd.ac.in
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Initialize time measurement
    tic
    yTrain = yTrain';
    yTest = yTest';
    % Check if labels are column vectors:
    if( (size(yTrain,2) > 1) || (size(yTest,2) > 1) )
        fprintf(2,'Please make sure that the label vectors are column vectors!\n');
        return;
    end

    
    % Check if labels are all +1 and -1:
    if( (length(sort(unique(yTrain))) ~= 2) || (length(sort(unique(yTest))) ~= 2) )
        fprintf(2,'Please make sure labels are all +1 and -1!\n');
        return;
    elseif( ((sum(sort(unique(yTrain)) == [-1,1]')) ~= 2) || ((sum(sort(unique(yTest)) == [-1,1]')) ~= 2) )
        fprintf(2,'Please make sure labels are all +1 and -1!\n');
        return;
    end
    
    
    % Define variables for training data
    N=size(xTrain,1);
    D=size(xTrain,2);

    
    % Define Kernel
    if(Kernel_Flag == 0)
        % Linear
        Kernel = @(x,y) ( (x*y'));
    elseif(Kernel_Flag == 1)
        % RBF
        Kernel = @(x,y) exp(-beta * norm(x-y)^2);    
    else
        fprintf(2,'Please enter valid Kernel Flag!\n');
        return;
    end
    
    
    %Normalize data: Make each feature Mean 0 and Variance 1
    for i = 1:D
        m(i) = mean(xTrain(:,i));
        s(i) = std(xTrain(:,i));
        if(s(i) == 0)
            xTrain(:,i) = (xTrain(:,i)-m(i));
            xTest(:,i) = xTest(:,i)-m(i);
        else
            xTrain(:,i) = (xTrain(:,i)-m(i))/s(i);
            xTest(:,i) = xTest(:,i)-m(i)/s(i);
        end
    end
    
    
    % Initialize variables for solving the MCM LP
    X = [randn(N,1);randn(1,1);randn(N,1);randn(1,1)];       %[lambda, b, q, h]
    f = [zeros(N,1);zeros(1,1);C*ones(N,1);1];    

    
    % Define Constraint matrix
    CM = zeros(N,N);
    for i=1:N
        for j=1:N
            CM(i,j) = yTrain(i) * Kernel(xTrain(i,:),xTrain(j,:));
        end
    end
    
    
    % Define inequality constraints
    %   [lambda,               b,          q,              h]
    A = [ CM        ,     yTrain,   eye(N,N),   -1*ones(N,1);
         -CM        ,    -yTrain,  -eye(N,N),     zeros(N,1);];
    B = [zeros(N,1);-1*ones(N,1);];


    % Define equality constraints
    Aeq = [];
    Beq = [];

    
    % Define bounds on variables
    %    [        lambda,      b,             q,     h]
    lb = [-inf*ones(N,1);   -inf;    zeros(N,1);     0;];
    ub = [ inf*ones(N,1);    inf; inf*ones(N,1);   inf;];

    
    % Options for solving the MCM LP: we use the simplex method
    options=optimset('display','none', 'Largescale', 'off', 'Simplex', 'on');

    
    % Solve the MCM Linear Program(LP) and store results
    [X,~,exitflag]  = linprog(f,A,B,Aeq,Beq, lb,ub, [], options);
    lambda = X(1:N,:);
    b = X(N + 1,:);
    q = X(N+1 + 1:N+1 + N,:);
    H = X(2*N+1 + 1,:);

    % Check if LP was solved to optimality
    if(exitflag ~=1)
        fprintf(2,'Linear Program was not solved to optimality. Check data or choose different parameters!\n');
        return;
    end
    
    
    % Define variable for storing in-sample prediction values
    yPredTrain = yTrain*0;
    
    
    % Compute in-sample predictions
    for i = 1:N
        sumj = b;
        for j = 1:N
            sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTrain(i, :));
        end
        yPredTrain(i) = sumj;
    end
    
    
    % Compute in-sample prediction accuracy
    trainAcc = sum(yPredTrain.*yTrain>0)/size(yTrain,1) * 100;

    
    % Define variables for storing out-of-sample prediction values
    Ntest = size(xTest, 1);
    yPredTest = yTest*0;

    
    % Compute out-of-sample predictions
    for i = 1:Ntest
        sumj = b;
        for j = 1:N
            sumj = sumj + lambda(j) * Kernel(xTrain(j, :), xTest(i, :));
        end
        yPredTest(i) = sumj;
    end
    
    
    % Compute out-of-sample accuracy
    testAcc = sum(yPredTest.*yTest>0)/size(yTest,1) * 100;    

    
    % Compute number of support vectors
    nsv = 0;
    for k = 1:N
        if(lambda(k)~=0)
            nsv = nsv +1;
        end
    end
    
    
    % Compute time taken
    time = toc;

    
    
    % Display result statistics
    fprintf(2, 'MCM\nTraining set accuracy: %f \t Test set accuracy: %f \t nsv: %d \t Time = %f \n', trainAcc, testAcc, nsv, time);

end
