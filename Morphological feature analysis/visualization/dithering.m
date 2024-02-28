function [x,y,Index,angl,ind1,ind2]=dithering(x,y)
%DITHERING is used to check whether the data points are in general position.
% If not, then a very small random number is added to each of
% the data points. Also the angles formed by pairs of data points are
% computed here. This file is used in the functions isodepth.m, halfmed.m
% and bagp.m.
%
% Required input arguments:
%            x : vector containing the first coordinates of all the data
%                points
%            y : vector containing the second coordinates of all the data
%                points

if not(length(x)==length(y))
    error('The vectors x and y must have the same length.')
end
if sum(isnan(x))>=1 || sum(isnan(y))>=1
    error('Missing values are not allowed')
end
n=length(x);
i=0;
blijf3=1;
dith=1;
xorig=x;
yorig=y;
while dith==1
    [xs,Index]=sort(x);
    ys=y(Index);
    dith=0;
    while blijf3==1
        blijf3=0;
        i=i+1;
        if (i+1)>n
            blijf3=2;
        else
            j=i+1;
            if not(xs(i)==xs(j))
                blijf3=1;
            else
                if ys(i)==ys(j)
                    dith=1;
                    blijf3=0;
                    % two datapoints coincide
                else
                    if ys(i)<ys(j)
                        even=Index(j);
                        Index(j)=Index(i);
                        Index(i)=even;
                    end
                    if (j+1)<=n
                        next=j+1;
                        if xs(i)==xs(next)
                            dith=1;
                            blijf3=0;
                            % three data points are collinear
                        else
                            blijf3=1;
                        end
                    end
                end
            end
        end
        if dith==1
            fac=1000000;
            ran=randn(n,2);
            x=xorig+ran(:,1)/fac;
            wx=x;
            y=yorig+ran(:,2)/fac;
            wy=y;
        else
            x=x;
            y=y;
        end
    end
    %
    if dith==0
        %
        % Compute all the angles formed by pairs of data points
        %
        m=0;
        for i=1:n
            rest=(i+1):n;
            spec=intersect(find(x(i)==x),rest);
            hoek(spec+m-i)=pi/2;
            spec=intersect(find(not(x(i)==x)),rest);
            hoek(spec+m-i)=atan((y(i)-y(spec))./(x(i)-x(spec)));
            p=m;
            m=m+n-i;
            if sum(hoek<=0)>=1
                spec=find(hoek<=0);
                hoek(spec)=hoek(spec)+pi;
            end
            ind1((p+1):m)=i;
            ind2((p+1):m)=rest;
        end
        %
        % Sort all the angles and permute ind1 and ind2 in the same way.
        %
        [angl,In]=sort(hoek);
        %
        ind1=ind1(In);
        ind2=ind2(In);
        %
        % Test wether any three datapoints are collinear.
        %
        ppp=diff(angl);
        k=find(ppp==0);
        %
        if sum(ind1(k)==ind1(k+1))>=1
            % There are 3 or more datapoints collineair.
            dith=1;
            fac=100000000;
            ran=randn(n,2);
            x=xorig+ran(:,1)/fac;
            wx=x;
            y=yorig+ran(:,2)/fac;
            wy=y;
        else
            x=x;
            y=y;
        end

    end
end

