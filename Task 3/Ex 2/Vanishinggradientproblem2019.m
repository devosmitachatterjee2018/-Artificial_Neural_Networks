clc; clear;
%%
% Name: Devosmita Chatterjee
% Assignment 3.1
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(2);
xTrain = xTrain-mean(xTrain,2);

epochs = 100;
eta = 0.01;
mB = 100;
p = size(xTrain,2);
batches = p/mB;

w1 = normrnd(0,1/sqrt(3072),20,size(xTrain,1));
theta1 = zeros(20,1);
w2 = normrnd(0,1/sqrt(20),20,20);
theta2 = zeros(20,1);
w3 = normrnd(0,1/sqrt(20),20,20);
theta3 = zeros(20,1);
w4 = normrnd(0,1/sqrt(20),20,20);
theta4 = zeros(20,1);
w5 = normrnd(0,1/sqrt(20),size(tTrain,1),20);
theta5 = zeros(size(tTrain,1),1);

energy_function = zeros(1,epochs);


epoch_err1 = zeros(1,epochs);
epoch_err2 = zeros(1,epochs);
epoch_err3 = zeros(1,epochs);
epoch_err4 = zeros(1,epochs);
epoch_err5 = zeros(1,epochs);
%%

for t = 1:epochs
    t
    
    p = 40000;
    rng(55)
    tmp = randperm(length(xTrain));
    xTrain =  xTrain(:,tmp);
    tTrain =  tTrain(:,tmp);
    
    out = zeros(10,40000);
    
    err1_f =0;
    err2_f =0;
    err3_f =0;
    err4_f =0;
    err5_f =0;
    
    for nbr = 1:batches
        nbr
        
        %tempw1=zeros(20,size(xTrain,1));
        %tempt1=zeros(20,1);
        
        %tempw2=zeros(20,20);
        %tempt2=zeros(20,1);
        
        %tempw3=zeros(20,20);
        %tempt3=zeros(20,1);
        
        %tempw4=zeros(20,20);
        %tempt4=zeros(20,1);
        
        %tempw5=zeros(20,size(tTrain,1));
        %tempt5=zeros(size(tTrain,1),1);
        
        del1 = 0;
        err1 = 0;
        del2 = 0;
        err2 = 0;
        del3 = 0;
        err3 = 0;
        del4 = 0;
        err4 = 0;
        del5 = 0;
        err5 = 0;
        
        
        
        for mu=(nbr-1)*mB+1:nbr*mB
            
            batch_xTrain = xTrain(:,mu);
            batch_tTrain = tTrain(:,mu);
            V0 = batch_xTrain;
            b1 = w1*V0-theta1;
            V1 =1./(1+exp(-b1));
            b2 = w2*V1-theta2;
            V2 =1./(1+exp(-b2));
            b3 = w3'*V2-theta3;
            V3 =1./(1+exp(-b3));
            b4 = w4*V3-theta4;
            V4 =1./(1+exp(-b4));
            b5 = w5*V4-theta5;
            V5 =1./(1+exp(-b5));
            for i=1:10
                out(i,mu) = V5(i);
            end
            
            error5=(batch_tTrain-V5).*V5.*(1-V5);
            error4=w5'*error5.*V4.*(1-V4);
            error3=w4'*error4.*V3.*(1-V3);
            error2=w3*error3.*V2.*(1-V2);
            error1=w2'*error2.*V1.*(1-V1);
            
            err1 = error1+err1;
            err2 = error2+err2;
            err3 = error3+err3;
            err4 = error4+err4;
            err5 = error5+err5;
            
            delta5 = error5*V1';
            del5 = delta5 +del5;
            delta4 = error4*V3';
            del4 = delta4 +del4;
            delta3 = error3*V2';
            del3 = delta3 +del3;
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
        
        tempw3=eta*del3';
        tempt3=-eta*err3;
        w3=w3+tempw3;
        theta3=theta3+tempt3;
        
        tempw4=eta*del4';
        tempt4=-eta*err4;
        w4=w4+tempw4;
        theta4=theta4+tempt4;
        
        tempw5=eta*del5;
        tempt5=-eta*err5;
        w5=w5+tempw5;
        theta5=theta5+tempt5;
        
        err1_f =err1+err1_f;
        err2_f =err2+err2_f;
        err3_f =err3+err3_f;
        err4_f =err4+err4_f;
        err5_f =err5+err5_f;
        
    end
    
    stored=0;
    for m = 1:40000
        stored = sum(abs(tTrain(:,m)-out(:,m)).^2)+stored;
    end
    H=stored/2;
    energy_function(t) = H;
    
    epoch_err1(t) = norm(err1_f);
    epoch_err2(t) = norm(err2_f);
    epoch_err3(t) = norm(err3_f);
    epoch_err4(t) = norm(err4_f);
    epoch_err5(t) = norm(err5_f);
    
end


%%
x = 1:100;

plot(x,energy_function,'k')

xlabel('Number of epochs')
ylabel('H')
title('Energy function plot')

%%
x = 1:100;

plot(x,epoch_err1,'k')
hold on
plot(x,epoch_err2,'b')
hold on
plot(x,epoch_err3,'m')
hold on
plot(x,epoch_err4,'c')
hold on
plot(x,epoch_err5,'r')

set(gca, 'YScale', 'log')
xlabel('Number of epochs')
ylabel('U^{(l)}')

legend('l = 1','l = 2','l = 3','l = 4','l = 5','Location','Best')
