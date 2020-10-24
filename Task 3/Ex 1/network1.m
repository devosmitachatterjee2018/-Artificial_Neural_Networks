function [C_Train,C_Valid,C_Test,lastepoch1] = network1(xTrain,tTrain,xValid,tValid,xTest,tTest)

epochs = 20;
eta = 0.1;
mB = 100;
p = size(xTrain,2);
batches = p/mB;

w = normrnd(0,1/sqrt(3072),size(tTrain,1),size(xTrain,1));
theta = zeros(size(tTrain,1),1);

w1=zeros(size(tTrain,1),size(xTrain,1),epochs);
theta1=zeros(size(tTrain,1),1,epochs);

C_Train = zeros(1,epochs);
C_Valid = zeros(1,epochs);

for t = 1:epochs
    t
    
    p = 40000;
    rng(55)
    tmp = randperm(length(xTrain));
    xTrain =  xTrain(:,tmp);
    tTrain =  tTrain(:,tmp);
    
    for nbr = 1:batches
        nbr
        
        tempw=zeros(size(tTrain,1),size(xTrain,1));
        tempt=zeros(size(tTrain,1),1);
        del = 0;
        err = 0;
        for mu=(nbr-1)*mB+1:nbr*mB
            batch_xTrain = xTrain(:,mu);
            batch_tTrain = tTrain(:,mu);
            V0 = batch_xTrain;
            b = w*V0-theta;
            V1 =1./(1+exp(-b));
            error=(batch_tTrain-V1).*V1.*(1-V1);
            err = error+err;
            delta = error*V0';
            del = delta +del;
        end
        
        tempw=eta*del;
        tempt=-eta*err;
        w=w+tempw;
        theta=theta+tempt;
    end
    
    %training set
    C_Terr = 0;
    for mu = 1:size(xTrain,2)
        V0_Train = xTrain(:,mu);
        b_Train = w*V0_Train-theta;
        V1_Train = 1./(1+exp(-b_Train));
        out_Train = zeros(size(tTrain,1),1);
        index_Train = find(V1_Train == max(V1_Train));
        out_Train(index_Train) = 1;
        C_Terr = (1/(2*size(xTrain,2)))*norm(tTrain(:,mu)-out_Train) + C_Terr;
    end
    
    %Validation set
    C_Verr = 0;
    for mu = 1:size(xValid,2)
        V0_Valid = xValid(:,mu);
        b_Valid = w*V0_Valid-theta;
        V1_Valid = (1+exp(-b_Valid)).^(-1);
        out_Valid = zeros(size(tValid,1),1);
        index_Valid = find(V1_Valid == max(V1_Valid));
        out_Valid(index_Valid) = 1;
        C_Verr = (1/(2*size(xValid,2)))*norm(tValid(:,mu)-out_Valid) + C_Verr;
        
    end
    %calculate classification errors
    C_Train(t) = C_Terr;
    C_Valid(t) = C_Verr;
    w1(:,:,t) = w;
    theta1(:,:,t) = theta;
end

index_min = find(C_Valid == min(C_Valid));
weight = w1(:,:,index_min(end));
threshold = theta1(:,:,index_min(end));
lastepoch1 = index_min(end);

%Test set
C_err = 0;
for mu = 1:size(xTest,2)
    V0_Test = xTest(:,mu);
    b_Test = weight*V0_Test-threshold;
    V1_Test = 1./(1+exp(-b_Test));
    out_Test = zeros(size(tTest,1),1);
    index_Test = find(V1_Test == max(V1_Test));
    out_Test(index_Test) = 1;
    C_err = (1/(2*size(xTest,2)))*norm(tTest(:,mu)-out_Test) + C_err;
end
%calculate classification errors
C_Test = C_err;

end