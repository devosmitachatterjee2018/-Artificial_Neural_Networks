function [C_Train,C_Valid,C_Test,lastepoch3] = network3(xTrain,tTrain,xValid,tValid,xTest,tTest)

epochs = 20;
eta = 0.1;
mB = 100;
p = size(xTrain,2);
batches = p/mB;

w1 = normrnd(0,1/sqrt(3072),50,size(xTrain,1));
theta1 = zeros(50,1);
w2 = normrnd(0,1/sqrt(50),size(tTrain,1),50);
theta2 = zeros(size(tTrain,1),1);

save_w1=zeros(50,size(xTrain,1),epochs);
save_theta1=zeros(50,1,epochs);
save_w2=zeros(size(tTrain,1),50,epochs);
save_theta2=zeros(size(tTrain,1),1,epochs);

C_Train = zeros(1,epochs);
C_Valid = zeros(1,epochs);

%%

for t = 1:epochs
    t
    
    p = 40000;
    rng(55)
    tmp = randperm(length(xTrain));
    xTrain =  xTrain(:,tmp);
    tTrain =  tTrain(:,tmp);
    
    
    for nbr = 1:batches
        nbr
        
        tempw1=zeros(50,size(xTrain,1));
        tempt1=zeros(50,1);
        
        tempw2=zeros(size(tTrain,1),50);
        tempt2=zeros(size(tTrain,1),1);
        
        del1 = 0;
        err1 = 0;
        del2 = 0;
        err2 = 0;
        for mu=(nbr-1)*mB+1:nbr*mB
            batch_xTrain = xTrain(:,mu);
            batch_tTrain = tTrain(:,mu);
            V0 = batch_xTrain;
            b1 = w1*V0-theta1;
            V1 =1./(1+exp(-b1));
            b2 = w2*V1-theta2;
            V2 =1./(1+exp(-b2));
            
            
            error2=(batch_tTrain-V2).*V2.*(1-V2);
            error1=w2'*error2.*V1.*(1-V1);
            
            err1 = error1+err1;
            err2 = error2+err2;
            
            delta2 = error2*V1';
            del2 = delta2 +del2;
            delta1 = error1*V0';
            del1 = delta1 +del1;
            
        end
        
        tempw1=eta*del1;
        tempt1=-eta*err1;
        w1=w1+tempw1;
        theta1=theta1+tempt1;
        
        tempw2=eta*del2;
        tempt2=-eta*err2;
        w2=w2+tempw2;
        theta2=theta2+tempt2;
    end
    
    
    
    
    %training set
    
    C_Terr = 0;
    for mu = 1:size(xTrain,2)
        V0_Train = xTrain(:,mu);
        b1_Train = w1*V0_Train-theta1;
        V1_Train =1./(1+exp(-b1_Train));
        b2_Train = w2*V1_Train-theta2;
        V2_Train =1./(1+exp(-b2_Train));
        out_Train = zeros(size(tTrain,1),1);
        index_Train = find(V2_Train == max(V2_Train));
        out_Train(index_Train) = 1;
        C_Terr = (1/(2*size(xTrain,2)))*norm(tTrain(:,mu)-out_Train) + C_Terr;
    end
    
    %Validation set
    C_Verr = 0;
    for mu = 1:size(xValid,2)
        V0_Valid = xValid(:,mu);
        b1_Valid = w1*V0_Valid-theta1;
        V1_Valid =1./(1+exp(-b1_Valid));
        b2_Valid = w2*V1_Valid-theta2;
        V2_Valid =1./(1+exp(-b2_Valid));
        out_Valid = zeros(size(tValid,1),1);
        index_Valid = find(V2_Valid == max(V2_Valid));
        out_Valid(index_Valid) = 1;
        C_Verr = (1/(2*size(xValid,2)))*norm(tValid(:,mu)-out_Valid) + C_Verr;
        
    end
    %calculate classification errors
    C_Train(t) = C_Terr;
    C_Valid(t) = C_Verr;
    save_w1(:,:,t) = w1;
    save_theta1(:,:,t) = theta1;
    save_w2(:,:,t) = w2;
    save_theta2(:,:,t) = theta2;
end

index_min = find(C_Valid == min(C_Valid));
weight1 = save_w1(:,:,index_min(end));
threshold1 = save_theta1(:,:,index_min(end));
weight2 = save_w2(:,:,index_min(end));
threshold2 = save_theta2(:,:,index_min(end));
lastepoch3 = index_min(end);
%%
%Test set
C_err = 0;
for mu = 1:size(xTest,2)
    V0_Test = xTest(:,mu);
    b1_Test = weight1*V0_Test-threshold1;
    V1_Test =1./(1+exp(-b1_Test));
    b2_Test = weight2*V1_Test-threshold2;
    V2_Test =1./(1+exp(-b2_Test));
    out_Test = zeros(size(tTest,1),1);
    index_Test = find(V2_Test == max(V2_Test));
    out_Test(index_Test) = 1;
    C_err = (1/(2*size(xTest,2)))*norm(tTest(:,mu)-out_Test) + C_err;
    
end
%calculate classification errors
C_Test = C_err;

end