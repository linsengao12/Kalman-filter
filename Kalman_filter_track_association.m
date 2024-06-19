% Track association method in dual receiver situation
% Based on KF-PDA

clc;
clear;
close all;

time=1000;% Monte Carlo times

% Parameters
T=4;  % Time interval
q1=0; % Process noise on x
q2=0; % Process noise on y
r=1;  % Measurement noise
F=[1 T 0 0;0 1 0 0;0 0 1 T;0 0 0 1]; % State transition matrix
C=[T^2/2 0;T 0;0 T^2/2;0 T]; % Process noise distribution matrix
H=[1 0 0 0;0 0 1 0]; % Measurement transition matrix
gama=5;
lamda=0.004; % Density of spurious measurement
len=9; % Total tracking steps
PD=0.8; % Detetion probability
PG=1-exp(-0.5*gama); % Gate probability

R=[r^2  0;0  r^2];
P(:,1:4)=[r^2 r^2/T 0 0;r^2/T 2*r^2/T^2 0 0;0 0 r^2 r^2/T;0 0 r^2/T 2*r^2/T^2];

x=0;  % Initial position on x
y=0;  % Initial position on y
vx=2; % Initial velocity on x
vy=2; % Initial position on y

% Coordinate system translation transformation paras
a1=100;
b1=200;
a2=10;
b2=20;
BH=[a1,a2;0,0;b1,b2;0,0];

count_num1=0;
count_num2=0;

a=0.05 ; % Significance level
nx=4; % Dimension of the state vector

for v=1:len+1
    dd(v,:)=chi2inv(1-a,nx*v);
end
d_th=chi2inv(1-a,nx);

for kk=1:time
    
    kk
    
    P(:,1:4)=[r r/T 0 0;r/T 2*r/T^2 0 0;0 0 r r/T;0 0 r/T 2*r/T^2];
    
    for m=1:2  % Dual-receiver
        clear xi yi
        X(:,1)=[x;vx;y;vy]+BH(:,m);% Coordinate transformation
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
            sq=pi*gama*sqrt(det(S(:,2*k-1:2*k))); % Gate
            nc=floor(10*sq*lamda+1); % Spurious measurement number
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
            bb(k)=lamda*sqrt(det(2*pi*S(:,2*k-1:2*k)))*(1-PD*PG)/PD;
            % Choosing the candidate measurements
            mk=0;
            pj=[];
            pjj=[];
            gv=[];
            for  i=1:(nc+1)
                pj(i)=vv(:,i)'*inv(S(:,2*k-1:2*k))*vv(:,i);
                if pj(i)<=gama
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

        
        % Relative position from two receivers
        xxZ(2*m-1,:)=XX(1,:);
        xxZ(2*m,:)=XX(3,:);
        zzZ(2*m-1,:)=ZZ(1,:);
        zzZ(2*m,:)=ZZ(2,:);
        xx1Z(2*m-1,:)=XX1(1,:);
        xx1Z(2*m,:)=XX1(3,:);
        
        % Velocity from two receivers
        xxV(2*m-1,:)=XX(2,:);
        xxV(2*m,:)=XX(4,:);
        xx1V(2*m-1,:)=XX1(2,:);
        xx1V(2*m,:)=XX1(4,:);
        
        % Covariance from two receivers
        ppZ(4*m-3,:)=PP(1,:);
        ppZ(4*m-2,:)=PP(2,:);
        ppZ(4*m-1,:)=PP(3,:);
        ppZ(4*m,:)=PP(4,:);
        
    end
    
    % Transform to the common coordinate system
    
    for j=1:len+1
        xxZ(1,j)=xxZ(1,j)-a1;
        xxZ(2,j)=xxZ(2,j)-b1;
        xxZ(3,j)=xxZ(3,j)-a2;
        xxZ(4,j)=xxZ(4,j)-b2;
        zzZ(1,j)=zzZ(1,j)-a1;
        zzZ(2,j)=zzZ(2,j)-b1;
        zzZ(3,j)=zzZ(3,j)-a2;
        zzZ(4,j)=zzZ(4,j)-b2;
    end
    
    
    for i=2:len+1
        t(1,i)=(1/sqrt(2*r))*(xxZ(1,i)-xxZ(3,i));
        t(2,i)=(1/sqrt(2*r))*(xxV(1,i)-xxV(3,i));
        t(3,i)=(1/sqrt(2*r))*(xxZ(2,i)-xxZ(4,i));
        t(4,i)=(1/sqrt(2*r))*(xxV(2,i)-xxV(4,i));
        cc(1:4,4*i-3:4*i)=ppZ(1:4,4*i-3:4*i)+ppZ(5:8,4*i-3:4*i);
        rc(i)=t(:,i)'*(inv(cc(1:4,4*i-3:4*i)))*t(:,i);
    end
    
    % Sequential track accosiation method
    rr=sum(rc);
    if rr<=dd(len+1)
        count_num1=count_num1+1;
    end
    
    % Double threshold accosiation method
    count_num3=0;
    for k=1:len+1
        if rc(k)<=d_th
            count_num3=count_num3+1;
        end
    end
    
    if count_num3>=8
        count_num2=count_num2+1;
    end
    
end

count_num1
count_num2

