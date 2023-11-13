%%%%% Classiffying all 3 cases %%%%
clc;
clear all;
close all;
cen=10;
load('centers.mat')
C= table2array(centers);
load('weights.mat')
w=table2array(centersS1) %Normal weights
%C=2*rand(cen,4)-1
%w=(2*rand(1,cen)-1)*0.01
wb=0.2; %Bias weight

n=0.1;
r=1.000109; %Width
yk(1)=0.1;

%Data Generation
load('datasample.mat')
T=table2array(annprojectS4);
for i=1:240
    x(1,i)=T(i,1);
      x(2,i)=T(i,2);
      x(3,i)=T(i,3);
      x(4,i)=T(i,4);
      yd(i)=x(4,i);
end

for epoch=1:165
    e=0;
    for i=1:240
        for k=1:cen
        d(k)=((x(1,i)-C(k,1))^2)+((x(2,i)-C(k,2))^2)+((x(3,i)-C(k,3))^2)+((x(4,i)-C(k,4))^2);
        z(k)=sqrt(d(k));
        end
        for k=1:cen
            V(k)=exp(((-1)*(z(k)^2))/((1)*(r^2)));
        end
        y(i)=0;
        for u=1:cen
        y(i)=y(i)+(V(u)*w(u))+wb;
        end
        for l=1:cen
            w(l)=w(l)+n*(yd(i)-y(i))*V(l);
        end
        wb=wb+n*(yd(i)-y(i));
        for q=1:cen
        for p=1:4
            C(q,p)=C(q,p)+n*(yd(i)-y(i))*w(q)*(V(q)/(r^2))*(x(p,i)-C(q,p));
        end
        end
        e=e+(0.001*((yd(i)-y(i))^2));
    end
    Er(epoch) = e;
    disp('epoch=');
    disp(epoch);
    disp('error=');
    disp(e);
    
end

sum=0;
for i=1:240
    sum=sum+(y(i)-yd(i))^2;
end
sum = sum/240;
mse_training=sum;

%Plotting Training Performance
figure,
plot(yd,'r--','LineWidth', 2), hold on, plot(y,'k--'), 
title('Training Results'); 
legend('Desired Output','Actual Network Output');
set(legend,'FontSize',11);
xlabel('Input vector index'), ylabel('Output');
%title('Tip Position Trajectory');
set(gca,'FontSize',12)
h_xlabel = get(gca,'XLabel');
set(h_xlabel,'FontSize',12); 
h_ylabel = get(gca,'YLabel');
set(h_ylabel,'FontSize',12); 

%Plotting Error
figure,
plot(Er,'m:','LineWidth', 2)
%title('Training Results (Error Plot)'); 
legend('Error Norm');
set(legend,'FontSize',11);
xlabel('Epocs'), ylabel('Error');
set(gca,'FontSize',12)
h_xlabel = get(gca,'XLabel');
set(h_xlabel,'FontSize',12); 
h_ylabel = get(gca,'YLabel');
set(h_ylabel,'FontSize',12);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Testing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('newtesting.mat')
S=table2array(testingdataS3);
for l=1:45
    x(1,l)=S(l,1);
      x(2,l)=S(l,2);
      x(3,l)=S(l,3);
      x(4,l)=S(l,4);
      yds(l)=x(4,l);

      for k=1:cen
        ds(k)=((x(1,l)-C(k,1))^2)+((x(2,l)-C(k,2))^2)+((x(3,l)-C(k,3))^2)+((x(4,l)-C(k,4))^2);
        zs(k)=sqrt(ds(k));
        end
        for k=1:cen
            Vs(k)=exp(((-1)*(zs(k)^2))/((1)*(r^2)));
        end
        ys(l)=0;
        for u=1:cen
        ys(l)=ys(l)+(Vs(u)*w(u))+wb;
        end
end


sum1=0;
for i=1:45
    sum1=sum1+(ys(i)-yds(i))^2;
end
sum1 = sum1/45;
mse_testing=sum1;

disp('mean square error in training = ');
disp(mse_training);
disp('mean square error in testing = ');
disp(mse_testing);

%Plotting Testing Performance
figure,
plot(yds,'r','LineWidth', 3), hold on, plot(ys,'k--'), 
%title('Testing Results'); 
legend('Desired Output','Actual Network Output');
set(legend,'FontSize',11);
xlabel('Input vector index'), ylabel('Output');
set(gca,'FontSize',12)
h_xlabel = get(gca,'XLabel');
set(h_xlabel,'FontSize',12); 
h_ylabel = get(gca,'YLabel');
set(h_ylabel,'FontSize',12); 

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

