% Kalman filter single-target tracking in 2-d Cartesian coordinate system

clc;
clear;
close all;

time=100; % Monte Carlo times

% Kalman parameters setting
T=4;  % Time interval
q1=0; % Process noise on x
q2=0; % Process noise on y
r=10; % Measurement noise
F=[1 T 0 0;0 1 0 0;0 0 1 T;0 0 0 1]; % State transition matrix
C=[T^2/2 0;T 0;0 T^2/2;0 T]; % Process noise distribution matrix
H=[1 0 0 0;0 0 1 0]; % Measurement transition matrix
len=99; % Total tracking steps
R=[r^2  0;0  r^2];
P(:,1:4)=[r^2 r^2/T 0 0;
    r^2/T 2*r^2/T^2 0 0;
    0 0 r^2 r^2/T;
    0 0 r^2/T 2*r^2/T^2];


x=0;  % Initial position on x
y=0;  % Initial position on y
vx=2; % Initial velocity on x
vy=2; % Initial position on y

% Sitimulate the real target track on x-y
for k=1:len+1
    ZS(1,k)=x+vx*k*T;
    ZS(2,k)=vx;
    ZS(3,k)=y+vy*k*T;
    ZS(4,k)=vy;
end

for t=1:time
    % Initialization
    X(:,1)=[x;vx;y;vy];
    noi1x=normrnd(0,q1,1,1);
    noi1y=normrnd(0,q2,1,1);
    noi2=normrnd(0,r,2,1);
    Z(:,1)=H*X(:,1)+noi2;
    X(:,2)=F*X(:,1)+C*[noi1x;noi1y];
    noi2=normrnd(0,r,2,1);
    Z(:,2)=H*X(:,2)+noi2;

    XX(:,1)=[Z(1,2);(Z(1,2)-Z(1,1))/T;Z(2,2);(Z(2,2)-Z(2,1))/T];
    
    % Kalman filter
    for k=1:len
        XX1(:,k+1)=F*XX(:,k); % State prediction XX1(k+1|k)=F*XX£¨k|k£©
        ZZ(:,k+1)=H*XX1(:,k+1);% Measurement prediction ZZ(k+1|k)=H*XX1(k+1|k)
        noi1x=normrnd(0,q1,1,1);
        noi1y=normrnd(0,q2,1,1);
        X(:,k+2)=F*X(:,k+1)+C*[noi1x;noi1y];
        noi2=normrnd(0,r,2,1);
        Z(:,k+2)=H*X(:,k+2)+noi2;
        U(:,k+1)=Z(:,k+2)-ZZ(:,k+1); % Innovation U(k)=Z(k+1)-ZZ(k+1|k)
        PP(:,4*k+1:4*k+4)=F*P(:,4*k-3:4*k)*F'+C*[q1 0;0 q2]*C'; % Covariance prediction PP(k+1|k)=F*P(k|k)*F'+Q(k)      
        S(:,2*k-1:2*k)=H*PP(:,4*k+1:4*k+4)*H'+R; % Innovation covariance S(k+1)=H*PP(k+1|k)*H'+R(k+1)
        K(:,2*k:2*k+1)=PP(:,4*k+1:4*k+4)*H'*inv(S(:,2*k-1:2*k)); % Kalman Gain K(k+1)=PP(k+1|k)*H'/S(k+1)
        XX(:,k+1)=XX1(:,k+1)+K(:,2*k:2*k+1)*U(:,k+1); % Update states XX(k+1|k+1)=XX1(k+1|k)+K(k+1)*U(k+1)
        P(:,4*k+1:4*k+4)=PP(:,4*k+1:4*k+4)-K(:,2*k:2*k+1)*S(:,2*k-1:2*k)*K(:,2*k:2*k+1)'; % Update covariance P(k+1|k+1)=PP(k+1|k)-K*S*K'
    end
    
    
% figure
% plot(ZS(1,:),ZS(3,:),'b',Z(1,:),Z(2,:),'g',XX(1,:),XX(3,:),'r')
% legend('Real track','Measurements','Kalman filter results')
% title('Kalman filter results')
% xlabel('x/m')
% ylabel('y/m')
% grid on
    
    
    for m=1:(len+1)
        err1(t,m)=(ZS(1,m)-XX(1,m))^2;
        err2(t,m)=(ZS(3,m)-XX(3,m))^2;
        err3(t,m)=(ZS(2,m)-XX(2,m))^2;
        err4(t,m)=(ZS(4,m)-XX(4,m))^2;
    end  
end

mean_err1=sqrt(mean(err1));
mean_err2=sqrt(mean(err2));
mean_err3=sqrt(mean(err3));
mean_err4=sqrt(mean(err4));


figure
plot(1:len+1,mean_err1,'r',1:len+1,mean_err2,'g','Linewidth',2)
title('Tracking error of position')
legend('x','y')
xlabel('Time (s)')
ylabel('Tracking error of position (m)')
grid on

figure
plot(1:len+1,mean_err3,'r',1:len+1,mean_err4,'g','Linewidth',2)
title('Tracking error of velocity')
legend('x','y')
xlabel('Time (s)')
ylabel('Tracking error of velocity£¨m/s£©')
grid on

