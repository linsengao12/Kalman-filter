% Kalman filter with probability density association (KF-PDA) single-target tracking in 2-d Cartesian coordinate system

clc;
clear;
close all;

time=100;% Monte Carlo times

% Kalman parameters setting
T=4;  % Time interval
q1=0; % Process noise on x
q2=0; % Process noise on y
r=5;  % Measurement noise
F=[1 T 0 0;0 1 0 0;0 0 1 T;0 0 0 1]; % State transition matrix
C=[T^2/2 0;T 0;0 T^2/2;0 T]; % Process noise distribution matrix
H=[1 0 0 0;0 0 1 0]; % Measurement transition matrix
gamma=9; 
lambda=0.001; % Density of spurious measurement
len=99; % Total tracking steps

x=0;  % Initial position on x
y=0;  % Initial position on y
vx=2; % Initial velocity on x
vy=2; % Initial position on y

PD=0.98; % Detetion probability
PG=1-exp(-0.5*gamma); % Gate probability

R=[r^2  0;0  r^2];
P(:,1:4)=[r^2 r^2/T 0 0;r^2/T 2*r^2/T^2 0 0;0 0 r^2 r^2/T;0 0 r^2/T 2*r^2/T^2];

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
    
    XX(:,1)=[Z(1,2);(Z(1,2)-Z(1,1))/T;Z(2,2);(Z(2,2)-Z(2,1))/T];%³õÊ¼×´Ì¬¹À¼Æ
    
    % Kalman filter
    for k=1:len
        XX1(:,k+1)=F*XX(:,k); % State prediction XX1(k+1|k)=F*XX£¨k|k£©
        ZZ(:,k+1)=H*XX1(:,k+1); % Measurement prediction ZZ(k+1|k)=H*XX1(k+1|k)
        noi1x=normrnd(0,q1,1,1);
        noi1y=normrnd(0,q2,1,1);
        X(:,k+2)=F*X(:,k+1)+C*[noi1x;noi1y];
        noi2=normrnd(0,r,2,1);
        Z(:,k+2)=H*X(:,k+2)+noi2;
        U(:,k+1)=Z(:,k+2)-ZZ(:,k+1); % Innovation U(k)=Z(k+1)-ZZ(k+1|k)
        PP(:,4*k+1:4*k+4)=F*P(:,4*k-3:4*k)*F'+C*[q1 0;0 q2]*C'; % Covariance prediction PP(k+1|k)=F*P(k|k)*F'+Q(k)
        S(:,2*k-1:2*k)=H*PP(:,4*k+1:4*k+4)*H'+R; % Innovation covariance S(k+1)=H*PP(k+1|k)*H'+R(k+1)
        K(:,2*k:2*k+1)=PP(:,4*k+1:4*k+4)*H'*inv(S(:,2*k-1:2*k)); % Kalman Gain K(k+1)=PP(k+1|k)*H'/S(k+1)
        % Adding spurious measurement
        sq=pi*gamma*sqrt(det(S(:,2*k-1:2*k))); % Gate
        nc=floor(10*sq*lambda+1); % Spurious measurement number
        q=sqrt(10*sq)/2;
        q=q/10;
        a=Z(1,k+2)-q;
        b=Z(1,k+2)+q;
        c=Z(2,k+2)-q;
        d=Z(2,k+2)+q;
        for j=1:nc
            xi(k,j)=a+(b-a)*rand(1);
            yi(k,j)=c+(d-c)*rand(1);
        end
        % All measurements
        vv=[];
        for jj=1:nc
            vv(1,jj)=xi(k,jj)-ZZ(1,k+1);
            vv(2,jj)=yi(k,jj)-ZZ(2,k+1);
        end
        vv(:,nc+1)=U(:,k+1); % Innovation sets
        bb(k)=lambda*sqrt(det(2*pi*S(:,2*k-1:2*k)))*(1-PD*PG)/PD;
        % Choosing the candidate measurements
        mk=0;
        pj=[];
        pjj=[];
        gv=[];
        for  i=1:(nc+1)
            pj(i)=vv(:,i)'*inv(S(:,2*k-1:2*k))*vv(:,i);
            if pj(i)<=gamma
                mk=mk+1;
                pjj(mk)=pj(i);
                gv(:,mk)=vv(:,i);
            end
        end
        e=[];
        beta=[];
        vvv=[];
        bv=[];
        if mk>0
            %ei
            for i=1:mk
                e(i)=exp(-pjj(i)/2);
            end
            %¦Âi
            for i=1:mk
                beta(i)=e(i)/(bb(k)+sum(e(1:mk)));
            end
            % Calculate composite innovation
            for i=1:mk
                vvv(:,i)=beta(i)*gv(:,i);
            end
            vvvv(:,1)=sum(vvv,2);
            % ¦Â0(k+1)
            beta0(k)=bb(k)/(bb(k)+sum(e(1:mk)));
            for i=1:mk
                bv(:,:,i)=beta(i)*gv(:,i)*gv(:,i)';
            end
            bvv=0;
            for i=1:mk
                bvv=bvv+bv(:,:,i);
            end
            XX(:,k+1)=XX1(:,k+1)+K(:,2*k:2*k+1)*vvvv; % Update states
            P(:,4*k+1:4*k+4)=beta0(k)*PP(:,4*k+1:4*k+4)+(1-beta0(k))*((eye(4)-K(:,2*k:2*k+1)*H)*PP(:,4*k+1:4*k+4))+K(:,2*k:2*k+1)*(bvv-vvvv*vvvv')*K(:,2*k:2*k+1)'; % Update covariance 
        else
            XX(:,k+1)=XX1(:,k+1);
            P(:,4*k+1:4*k+4)= PP(:,4*k+1:4*k+4);
        end
    end
    
%         figure
%         plot(xi(1,1),yi(1,1),'k.',ZS(1,:),ZS(3,:),'bx')
%         hold on
%         plot(Z(1,:),Z(2,:),'g','LineWidth',2)
%         hold on
%         plot(XX(1,:),XX(3,:),'r.-')
%         legend('Measurements','Real tracks','Measurements from target','Tracking result')
%         title('PDAF tracking result')
%         xlabel('x (m)')
%         ylabel('y (m)')
%         grid on
%         for i=1:len
%             for j=1:length(xi(i,:))
%                 if xi(i,j)~=0
%                     hold on
%                     plot(xi,yi,'k.')
%                 end
%             end
%         end
    
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

