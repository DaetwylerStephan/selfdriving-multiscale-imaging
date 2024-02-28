%-----------
function [kount,ADK,empty]=isodepth(x,y,d,varargin)

% ISODEPTH is an algoritm that computes the depth region of a bivariate dataset
% corresponding to depth d.
% First, we have to check whether the data points are in general position. If not,
% a very small random number is added to each of the data points until the
% dataset comes in general position. All this is done in the m-file dithering.
% Then all special k-dividers must be found. The coordinates of the vertices
% of the depth region we are looking for are intersection points of these
% special k-dividers. So, consequently, every intersection point in turn has
% to be tested, for example by computing its depth (see halfspacedepth.m),
% to check whether it is a vertex of the depth region.
%
% The ISODEPTH algoritm is described in:
%    Ruts, I., Rousseeuw, P.J. (1996),
%    "Computing depth contours of bivariate point clouds",
%    Computational Statistics and Data Analysis, 23, 153-168.
%
% Required input arguments:
%            x : vector containing the first coordinates of all the data
%                points
%            y : vector containing the second coordinates of all the data
%                points
%            d : the depth of which the depth region has to be constructed
%
%
% I/O: [kount, ADK, empty]= isodepth(x,y,d);
%
% The output of isodepth is given by
%
%        kount : the total number of vertices of the depth region
%        ADK   : the coordinates of the vertices of the depth region
%        empty : logical value (1 if the depth region is empty, 0 if not)
%
% This function is part of the Matlab Library for Robust Analysis,
% available at:
%              http://wis.kuleuven.be/stat/robust.html
%
% Last Update: 29/04/2005

n=length(x);
eps=0.0000001;
%
% Checking input
%
if length(x)==1
    error('x is not a vector')
elseif not(length(x)==length(y))
    error('The vectors x and y must have the same length.')
end
if sum(isnan(x))>=1 || sum(isnan(y))>=1
    error('Missing values are not allowed')
end
if sum(x==x(1))==n
    error('All data points lie on a vertical line.')
elseif sum(y==y(1))==n
    error('All data points lie on a horizontal line.')
else
    R=corrcoef(x,y);
    if abs(R(1,2))==1
        error('All data points are collineair.')
    end
end
%
% Check whether the data is in general position. If not, add a very small random
% number to each of the data points.
%
[x,y, Index, angl, ind1,ind2]=dithering(x,y);

%
% Main part
%
if (n==1)&&(d==1)
    kount=n;
    ADK=[x,y];
    empty=0;
    return
end
%
if (d>floor(n/2))
    kount=0;
    ADK=0;
    empty=1;
    return
end
%
if n<=3
    kount=n;
    ADK=[x,y];
    empty=1;
    return
end
%
nrank(Index)=(1:n);
%
% Let the line rotate from zero to angle(1)
%
ncirq=Index;
kount=1;
halt=0;
M=length(angl);
if angl(1)>(pi/2)
    L=1;
    D1=ind1(L);
    IV1=nrank(D1);
    D2=ind2(L);
    IV2=nrank(D2);
    IV=ncirq(IV1);
    ncirq(IV1)=ncirq(IV2);
    ncirq(IV2)=IV;
    IV=IV1;
    nrank(D1)=IV2;
    nrank(D2)=IV;
    %
    if ((IV1==d) && (IV2==(d+1)))||((IV2==d) && (IV1==(d+1)))||((IV1==(n-d)) && (IV2==(n-d+1)))||((IV2==(n-d)) && (IV1==(n-d+1)))
        if angl(L)<(pi/2)
            dum=angl(L)+(pi/2);
        else
            dum=angl(L)-(pi/2);
        end
        if (IV1==d && IV2==(d+1))||(IV2==d && IV1==(d+1))
            if dum<=(pi/2)
                alfa(kount)=angl(L)+pi;
            else
                alfa(kount)=angl(L);
            end
        end
        if or((IV1==(n-d) && IV2==(n-d+1)),(IV2==(n-d) && IV1==(n-d+1)))
            if dum<=(pi/2)
                alfa(kount)=angl(L);
            else
                alfa(kount)=angl(L)+pi;
            end
        end
        kand1(kount)=ind1(L);
        kand2(kount)=ind2(L);
        D(kount)=sin(alfa(kount))*x(ind1(L))-cos(alfa(kount))*y(ind1(L));
        kount=kount+1;
    end
    halt=1;
end
%
L=2;
stay=1;
% jflag keeps track of which angle we have to test next
while stay==1
    stay=0;
    kontrol=0;
    if (pi<=(angl(L)+(pi/2))) && ((angl(L)-(pi/2))< angl(1))
        D1=ind1(L);
        IV1=nrank(D1);
        D2=ind2(L);
        IV2=nrank(D2);
        IV=ncirq(IV1);
        ncirq(IV1)=ncirq(IV2);
        ncirq(IV2)=IV;
        IV=IV1;
        nrank(D1)=IV2;
        nrank(D2)=IV;
        %
        if ((IV1==d) && (IV2==(d+1)))||((IV2==d) && (IV1==(d+1)))||((IV1==(n-d)) && (IV2==(n-d+1)))||((IV2==(n-d)) && (IV1==(n-d+1)))
            if angl(L)<(pi/2)
                dum=angl(L)+(pi/2);
            else
                dum=angl(L)-(pi/2);
            end
            if (IV1==d && IV2==(d+1))||(IV2==d && IV1==(d+1))
                if dum<=(pi/2)
                    alfa(kount)=angl(L)+pi;
                else
                    alfa(kount)=angl(L);
                end
            end
            if or((IV1==(n-d) && IV2==(n-d+1)),(IV2==(n-d) && IV1==(n-d+1)))
                if dum<=(pi/2)
                    alfa(kount)=angl(L);
                else
                    alfa(kount)=angl(L)+pi;
                end
            end
            kand1(kount)=ind1(L);
            kand2(kount)=ind2(L);
            D(kount)=sin(alfa(kount))*x(ind1(L))-cos(alfa(kount))*y(ind1(L));
            kount=kount+1;
        end
        kontrol=1;
    end
    L=L+1;
    if kontrol==1
        halt=1;
    end
    if (L==(M+1)) && (kontrol==1)
        jflag=1;
        stay=2;
    end
    if not(stay==2)
        if ((halt==1)&&(kontrol==0))||(L==(M+1))
            stay=3;
        else
            stay=1;
        end
    end
end
if not(stay==2)
    if (L>1)
        jflag=L-1;
    else
        jflag=M;
    end
end
%
halt2=0;
if not(stay==2)
    J=0;
    %
    % If the first switch didnt occur between 0 and the angle angl(1) look for it
    % between the following angles.
    %
    stay2=1;
    if (L==M+1) && (kontrol==0)
        halt=0;
        halt2=0;
        J=J+1;
        if J==(M+1)
            J=1;
        end
        L=J+1;
        if L==(M+1)
            L=1;
        end
        while stay2==1
            stay2=0;
            kontrol=0;
            if (angl(L)+pi/2)<pi
                ang1=angl(L)+pi/2;
            else
                ang1=angl(L)-pi/2;
            end
            if J==M
                jj=1;
                if halt2==0
                    angl(1)=angl(1)+pi;
                end
            else
                jj=J+1;
            end
            if (angl(J)<=ang1) && (ang1<angl(jj))
                if angl(1)>pi
                    angl(1)=angl(1)-pi;
                end
                D1=ind1(L);
                IV1=nrank(D1);
                D2=ind2(L);
                IV2=nrank(D2);
                IV=ncirq(IV1);
                ncirq(IV1)=ncirq(IV2);
                ncirq(IV2)=IV;
                IV=IV1;
                nrank(D1)=IV2;
                nrank(D2)=IV;
                %
                if ((IV1==d) && (IV2==(d+1)))||((IV2==d) && (IV1==(d+1)))||((IV1==(n-d)) && (IV2==(n-d+1)))||((IV2==(n-d)) && (IV1==(n-d+1)))
                    if angl(L)<(pi/2)
                        dum=angl(L)+(pi/2);
                    else
                        dum=angl(L)-(pi/2);
                    end
                    if (IV1==d && IV2==(d+1))||(IV2==d && IV1==(d+1))
                        if dum<=(pi/2)
                            alfa(kount)=angl(L)+pi;
                        else
                            alfa(kount)=angl(L);
                        end
                    end
                    if or((IV1==(n-d) && IV2==(n-d+1)),(IV2==(n-d) && IV1==(n-d+1)))
                        if dum<=(pi/2)
                            alfa(kount)=angl(L);
                        else
                            alfa(kount)=angl(L)+pi;
                        end
                    end
                    kand1(kount)=ind1(L);
                    kand2(kount)=ind2(L);
                    D(kount)=sin(alfa(kount))*x(ind1(L))-cos(alfa(kount))*y(ind1(L));
                    kount=kount+1;
                end
                kontrol=1;
            end
            if angl(1)>pi
                angl(1)=angl(1)-pi;
            end
            if L==M
                L=1;
            else
                L=L+1;
            end
            if kontrol==1
                halt=1;
            end
            if (halt==1)&&(kontrol==0)
                if halt2==1
                    stay2=2;
                end
                if not(stay2==2)
                    if L>1
                        jflag=L-1;
                    else
                        jflag=M;
                    end
                    stay2=0;
                end
            else
                if L==jj
                    if jj==1
                        halt2=1;
                    end
                    J=J+1;
                    if J==(M+1)
                        J=1;
                    end
                    L=J+1;
                    if L==(M+1)
                        L=1;
                    end
                    stay2=1;
                else
                    stay2=1;
                end
            end
        end
    end
end
%
if not(stay2==2)
    %
    % The first switch has occurred. Now start looking for the next ones,
    % between the following angles.
    %
    for i=(J+1):(M-1)
        L=jflag;
        stay=1;
        while stay==1
            stay=0;
            kontrol=0;
            if ((angl(L)+pi/2)<pi)
                ang1=angl(L)+pi/2;
            else
                ang1=angl(L)-pi/2;
            end
            if (angl(i)<=ang1)&&(ang1<angl(i+1))
                D1=ind1(L);
                IV1=nrank(D1);
                D2=ind2(L);
                IV2=nrank(D2);
                IV=ncirq(IV1);
                ncirq(IV1)=ncirq(IV2);
                ncirq(IV2)=IV;
                IV=IV1;
                nrank(D1)=IV2;
                nrank(D2)=IV;
                %
                if ((IV1==d) && (IV2==(d+1)))||((IV2==d) && (IV1==(d+1)))||((IV1==(n-d)) && (IV2==(n-d+1)))||((IV2==(n-d)) && (IV1==(n-d+1)))
                    if angl(L)<(pi/2)
                        dum=angl(L)+(pi/2);
                    else
                        dum=angl(L)-(pi/2);
                    end
                    if (IV1==d && IV2==(d+1))||(IV2==d && IV1==(d+1))
                        if dum<=(pi/2)
                            alfa(kount)=angl(L)+pi;
                        else
                            alfa(kount)=angl(L);
                        end
                    end
                    if or((IV1==(n-d) && IV2==(n-d+1)),(IV2==(n-d) && IV1==(n-d+1)))
                        if dum<=(pi/2)
                            alfa(kount)=angl(L);
                        else
                            alfa(kount)=angl(L)+pi;
                        end
                    end
                    kand1(kount)=ind1(L);
                    kand2(kount)=ind2(L);
                    D(kount)=sin(alfa(kount))*x(ind1(L))-cos(alfa(kount))*y(ind1(L));
                    kount=kount+1;
                end
                kontrol=1;
            end
            if kontrol==0
                jflag=L;
            else
                if not(L==M)
                    L=L+1;
                else
                    L=1;
                end
                stay=1;
            end
        end
    end
    L=jflag;
    %
    % Finally, look for necessary switches between the last angle and zero.
    %
    stay=1;
    while stay==1
        kontrol=0;
        stay=0;
        if (angl(L)+pi/2)<pi
            ang1=angl(L)+pi/2;
        else
            ang1=angl(L)-pi/2;
        end
        if (angl(M)<=ang1)&&(ang1<pi)
            D1=ind1(L);
            IV1=nrank(D1);
            D2=ind2(L);
            IV2=nrank(D2);
            IV=ncirq(IV1);
            ncirq(IV1)=ncirq(IV2);
            ncirq(IV2)=IV;
            IV=IV1;
            nrank(D1)=IV2;
            nrank(D2)=IV;
            %
            if ((IV1==d) && (IV2==(d+1)))||((IV2==d) && (IV1==(d+1)))||((IV1==(n-d)) && (IV2==(n-d+1)))||((IV2==(n-d)) && (IV1==(n-d+1)))
                if angl(L)<(pi/2)
                    dum=angl(L)+(pi/2);
                else
                    dum=angl(L)-(pi/2);
                end
                if (IV1==d && IV2==(d+1))||(IV2==d && IV1==(d+1))
                    if dum<=(pi/2)
                        alfa(kount)=angl(L)+pi;
                    else
                        alfa(kount)=angl(L);
                    end
                end
                if or((IV1==(n-d) && IV2==(n-d+1)),(IV2==(n-d) && IV1==(n-d+1)))
                    if dum<=(pi/2)
                        alfa(kount)=angl(L);
                    else
                        alfa(kount)=angl(L)+pi;
                    end
                end
                kand1(kount)=ind1(L);
                kand2(kount)=ind2(L);
                D(kount)=sin(alfa(kount))*x(ind1(L))-cos(alfa(kount))*y(ind1(L));
                kount=kount+1;
            end
            kontrol=1;
        end
        if kontrol==1
            if not(L==M)
                L=L+1;
            else
                L=1;
            end
            stay=1;
        end
    end
end
num=kount-1; % num is the total number of special k-dividers
%
% Sort the num special k-dividers. Permute kand1, kand2 and D in the same
% way.
%
[alfa,In]=sort(alfa);
kand1=kand1(In);
kand2=kand2(In);
D=D(In);
%
IW1=1;
IW2=2;
Jfull=0;
NDK=0;
stay2=1;
while stay2==1
    stay2=0;
    ndata=0;
    %
    % Compute the intersection point.
    %
    while abs(-sin(alfa(IW2))*cos(alfa(IW1))+sin(alfa(IW1))*cos(alfa(IW2)))<eps
        IW2=IW2+1;
        ndata=0;
        if IW2==(num+1)
            IW2=1;
        end
    end
    %
    xcord=(cos(alfa(IW2))*D(IW1)-cos(alfa(IW1))*D(IW2))/(-sin(alfa(IW2))*cos(alfa(IW1))+sin(alfa(IW1))*cos(alfa(IW2)));
    ycord=(-sin(alfa(IW2))*D(IW1)+sin(alfa(IW1))*D(IW2))/(-sin(alfa(IW1))*cos(alfa(IW2))+sin(alfa(IW2))*cos(alfa(IW1)));
    %
    % Test whether the intersection point is a data point. If so,
    % adjust IW1 and IW2.
    %
    if or(kand1(IW1)==kand1(IW2),kand1(IW1)==kand2(IW2))
        ndata=kand1(IW1);
    end
    if or(kand2(IW1)==kand1(IW2),kand2(IW1)==kand2(IW2))
        ndata=kand2(IW1);
    end
    if not(ndata==0)
        iv=0;
        stay=1;
        while stay==1
            stay=0;
            next=IW2+1;
            iv=iv+1;
            if next==(num+1)
                next=1;
            end
            if not(next==IW1)
                if or(ndata==kand1(next),ndata==kand2(next))
                    IW2=IW2+1;
                    if (IW2==(num+1))
                        IW2=1;
                    end
                    stay=1;
                end
            end
        end
        if iv==(num-1)
            kount=1;
            ADK=[x(ndata),y(ndata)];
            empty=0;
            return
        end
    end
    if IW2==num
        kon=1;
    else
        kon=IW2+1;
    end
    if kon==IW1
        kon=kon+1;
    end
    if kon==(num+1)
        kon=1;
    end
    %
    % Test whether the intersection point lies to the left of the special
    % k-divider which corresponds to alfa(kon). If so, compute its depth.
    %
    stay3=1;
    stay4=1;
    if (sin(alfa(kon))*xcord-cos(alfa(kon))*ycord-D(kon))<=eps
        hdep1=halfspacedepth(xcord,ycord,x,y);
        if hdep1==d
            NDK=1;
        else
            hdep2=halfspacedepth(xcord-0.000001,ycord-0.000001,x,y);
            hdep3=halfspacedepth(xcord+0.000001,ycord+0.000001,x,y);
            hdep4=halfspacedepth(xcord-0.000001,ycord+0.000001,x,y);
            hdep5=halfspacedepth(xcord+0.000001,ycord-0.000001,x,y);
            hdepvector=[hdep1;hdep2;hdep3;hdep4;hdep5];
            if (NDK==0)&&(sum(hdepvector>=d)>=1)
                NDK=1;
            end
            if (hdep1<d)&&(hdep2<d)&&(hdep3<d)&&(hdep4<d)&&(hdep5<d)&&(NDK==1)
                %
                % The intersection point is not the correct one, try the next
                % special k-divider.
                %
                IW2=IW2+1;
                if IW2==(num+1)
                    IW2=1;
                end
                stay2=1;
            end
        end
        if not(stay2==1)
            %
            % Store IW2 and IW1 in kornr. If kornr has already been filled,
            % check wether we have encountered this intersection point before.
            %
            if (IW2>IW1)&&(Jfull==0)
                kornr(IW1:(IW2-1),1)=kand1(IW1);
                kornr(IW1:(IW2-1),2)=kand2(IW1);
                kornr(IW1:(IW2-1),3)=kand1(IW2);
                kornr(IW1:(IW2-1),4)=kand2(IW2);
            else
                if IW2>IW1
                    i=IW1;
                    stay3=1;
                    while stay3==1
                        if (kornr(i,1)==kand1(IW1))&&(kornr(i,2)==kand2(IW1))&&(kornr(i,3)==kand1(IW2))&&(kornr(i,4)==kand2(IW2))
                            stay3=0;
                        else
                            m1=(y(kornr(i,2))-y(kornr(i,1)))/(x(kornr(i,2))-x(kornr(i,1)));
                            m2=(y(kornr(i,4))-y(kornr(i,3)))/(x(kornr(i,4))-x(kornr(i,3)));
                            if not(m1==m2)
                                xcord1=(m1*x(kornr(i,1))-y(kornr(i,1))-m2*x(kornr(i,3))-y(kornr(i,3)))/(m1-m2);
                                ycord1=(m2*(m1*x(kornr(i,1))-y(kornr(i,1)))-m1*(m2*x(kornr(i,3))-y(kornr(i,3))))/(m1-m2);
                            end
                            if (abs(xcord1-xcord)<eps)&&(abs(ycord1-ycord)<eps)
                                stay3=0;
                            end
                            if stay3==1
                                kornr(i,1)=kand1(IW1);
                                kornr(i,2)=kand2(IW1);
                                kornr(i,3)=kand1(IW2);
                                kornr(i,4)=kand2(IW2);
                            end
                        end
                        if stay3==1
                            i=i+1;
                            if i==IW2
                                stay3=2;
                                i=i-1;
                            end
                        end
                    end
                else
                    Jfull=1;
                    kornr(IW1:num,1)=kand1(IW1);
                    kornr(IW1:num,2)=kand2(IW1);
                    kornr(IW1:num,3)=kand1(IW2);
                    kornr(IW1:num,4)=kand2(IW2);
                    i=1;
                    stay4=1;
                    if IW2==1
                        stay4=3;
                    end
                    while stay4==1
                        if (kornr(i,1)==kand1(IW1))&&(kornr(i,2)==kand2(IW1))&&(kornr(i,3)==kand1(IW2))&&(kornr(i,4)==kand2(IW2))
                            stay4=0;
                        else
                            m1=(y(kornr(i,2))-y(kornr(i,1)))/(x(kornr(i,2))-x(kornr(i,1)));
                            warning off MATLAB:divideByZero
                            m2=(y(kornr(i,4))-y(kornr(i,3)))/(x(kornr(i,4))-x(kornr(i,3)));
                            warning off MATLAB:divideByZero
                            if not(m1==m2)
                                xcord1=(m1*x(kornr(i,1))-y(kornr(i,1))-m2*x(kornr(i,3))-y(kornr(i,3)))/(m1-m2);
                                ycord1=(m2*(m1*x(kornr(i,1))-y(kornr(i,1)))-m1*(m2*x(kornr(i,3))-y(kornr(i,3))))/(m1-m2);
                            end
                            if (abs(xcord1-xcord)<=eps)&&(abs(ycord1-ycord)<=eps)
                                stay4=0;
                            end
                            if stay4==1
                                kornr(i,1)=kand1(IW1);
                                kornr(i,2)=kand2(IW1);
                                kornr(i,3)=kand1(IW2);
                                kornr(i,4)=kand2(IW2);
                            end
                        end
                        if stay4==1
                            i=i+1;
                            if i==IW2
                                i=i-1;
                                stay4=2;
                            end
                        end
                    end
                end
            end
        end
    elseif (stay3>0)&&(stay4>0)&&not(stay2==1)
        %
        % The intersection point is not the correct one, try the next
        % special k-divider.
        %
        IW2=IW2+1;
        if IW2==(num+1)
            IW2=1;
        end
        stay2=1;
    end
    %
    % Look for the next vertex of the convex figure.
    %
    if (stay3>0)&&(stay4>0)&&not(stay2==1)
        IW1=IW2;
        IW2=IW2+1;
        if IW2==(num+1)
            IW2=1;
        end
        stay2=1;
    end
end
%
% Scan kornr and ascribe the coordinates of the vertices to the variable
% ADK.
%
kount=0;
%
if NDK==0
    %
    % The requested depth region is empty
    %
    ADK=0;
    empty=1;
    return
else
    empty=0;
end
%
i=1;
E1=y(kornr(i,2))-y(kornr(i,1));
F1=x(kornr(i,1))-x(kornr(i,2));
G1=x(kornr(i,1))*(y(kornr(i,2))-y(kornr(i,1)))-y(kornr(i,1))*(x(kornr(i,2))-x(kornr(i,1)));
E2=y(kornr(i,4))-y(kornr(i,3));
F2=x(kornr(i,3))-x(kornr(i,4));
G2=x(kornr(i,3))*(y(kornr(i,4))-y(kornr(i,3)))-y(kornr(i,3))*(x(kornr(i,4))-x(kornr(i,3)));
xcord(i)=(-F2*G1+F1*G2)/(E2*F1-E1*F2);
ycord(i)=(-E2*G1+E1*G2)/(E1*F2-E2*F1);
DK(i,:)=[xcord(i),ycord(i)];
Juisteind(i)=i;
xcord1=xcord(i);
ycord1=ycord(i);
xcordp=xcord(i);
ycordp=ycord(i);
kount=kount+1;
i=i+1;
%
while not(i==num+1)
    if (kornr(i,1)==kornr(i-1,1))&&(kornr(i,2)==kornr(i-1,2))&&(kornr(i,3)==kornr(i-1,3))&&(kornr(i,4)==kornr(i-1,4))
        i=i+1;
    else
        if (kornr(i,1)==kornr(1,1))&&(kornr(i,2)==kornr(1,2))&&(kornr(i,3)==kornr(1,3))&&(kornr(i,4)==kornr(1,4))
            pp=find(not(Juisteind==0));
            ADK=DK(pp,:);
            empty=0;
            return
        else
            E1=y(kornr(i,2))-y(kornr(i,1));
            F1=x(kornr(i,1))-x(kornr(i,2));
            G1=x(kornr(i,1))*(y(kornr(i,2))-y(kornr(i,1)))-y(kornr(i,1))*(x(kornr(i,2))-x(kornr(i,1)));
            E2=y(kornr(i,4))-y(kornr(i,3));
            F2=x(kornr(i,3))-x(kornr(i,4));
            G2=x(kornr(i,3))*(y(kornr(i,4))-y(kornr(i,3)))-y(kornr(i,3))*(x(kornr(i,4))-x(kornr(i,3)));
            xcord(i)=(-F2*G1+F1*G2)/(E2*F1-E1*F2);
            ycord(i)=(-E2*G1+E1*G2)/(E1*F2-E2*F1);
            if ((abs(xcord(i)-xcordp)<eps)&&(abs(ycord(i)-ycordp)<eps))||((abs(xcord(i)-xcord1)<eps)&&(abs(ycord(i)-ycord1)<eps))
                i=i+1;
            else
                xcordp=xcord(i);
                ycordp=ycord(i);
                DK(i,:)=[xcord(i),ycord(i)];
                Juisteind(i)=i;
                kount=kount+1;
                i=i+1;
            end
        end
    end
end
%
% Delete all the empty spaces in the matrix DK. The result is ADK.
%
pp=find(not(Juisteind==0));
ADK=DK(pp,:);
