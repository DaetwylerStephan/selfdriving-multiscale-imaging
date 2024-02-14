
function [tukmed]=halfmed(x,y,varargin)

% HALFMED is an algoritm that computes the Tukey median of a two-dimensional
% dataset. First, we have to check whether the data points are in general position.
% If not, a random number is added to each of the data points until the dataset
% comes in general position. All this is done in the m-file dithering. Then
% the deepest depth region is constructed. One can accelerate this search
% by giving the optional argument kstar, which is usually the maximal halfspace
% depth of the data points. This advantage is used in the functie bagp.m to
% quicken the computations. The Tukey median is the center of gravity of
% this deepest depth region.
%
% The HALFMED algoritm is described in:
%    Rousseeuw, P.J., Ruts, I. (1998),
%    "Constructing the bivariate Tukey median",
%    Statistica Sinica, 8, 827-839.
%
% Required input arguments:
%            x : vector containing the first coordinates of all the data
%                points
%            y : vector containing the second coordinates of all the data
%                points
%
% Optional input argument:
%        kstar : an integer between ceil(n/3) and floor(n/2)
%                One can also use the maximum halfspace depth of the data
%                points. (default = 0)
%
%
% I/O: result=halfmed(x,y,'kstar',0);
%  The name of the input arguments needs to be followed by their value.
%
% The output of halfmed is given by
%
%       result : the coordinates of the Tukey median
%
% This function is part of the Matlab Library for Robust Analysis,
% available at:
%              http://wis.kuleuven.be/stat/robust.html
%
% Last Update: 29/04/2005

xsum=0;
ysum=0;
tukmed=0;
n=length(x);
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
% Looking for optional argument
%
if nargin>2
    if strcmp(varargin{1},'kstar')
        kstar=varargin{2};
        if length(kstar)>1
            error('kstar must be a number, not a vector.')
        end
    else
        error('Only kstar can be provided as an optional argument')
    end
else
    kstar=0;
end
if kstar>=floor(n/2)
    error('kstar must be smaller than floor(n/2) because the depth region corresponding with kstar is empty')
end
%
% Check whether the data are in general position. If not, add a very small random
% number to each of the data points.
%
[x,y, Index, angl, ind1,ind2]=dithering(x,y);
%
%Calculation of the Tukey median
%
if n<=3
    xsum=sum(x);
    ysum=sum(y);
    tukmed=[xsum/n,ysum/n];
    return
end
%
if kstar==0
    ib=ceil(n/3);
else
    ib=kstar;
end
ie=floor(n/2);
stay=1;
while stay==1
    le=ie-ib;
    if le<0
        le=0;
    end
    if le==0
        stay=0;
    end
    if stay==1
        [kou,dk,empty]=isodepth(x,y,ib+ceil(le/2));
        if empty==1
            ie=ib+ceil(le/2);
        end
        if empty==0
            ib=ib+ceil(le/2);
        end
        if le==1
            stay=0;
        end
    end
end
[kount,DK,empty]=isodepth(x,y,ib);
xsum=sum(DK(:,1));
if not(DK==0)
    ysum=sum(DK(:,2));
else
    ysum=0;
end
wx=DK(:,1);
if not(DK==0)
    wy=DK(:,2);
else
    wy=0;
end
%
% The maximal depth is now ib.
%
%
% Calculation of the center of gravity
%
som=0;
tukmed=0;
if kount>1
    wx=wx-(xsum/kount);
    wy=wy-(ysum/kount);
    for i=1:(kount-1)
        som=som+abs(wx(i)*wy(i+1)-wx(i+1)*wy(i));
        tukmed=tukmed+[(wx(i)+wx(i+1))*abs(wx(i)*wy(i+1)-wx(i+1)*wy(i)),(wy(i)+wy(i+1))*abs(wx(i)*wy(i+1)-wx(i+1)*wy(i))];
    end
    som=som+abs(wx(kount)*wy(1)-wx(1)*wy(kount));
    tukmed=tukmed+[(wx(kount)+wx(1))*abs(wx(kount)*wy(1)-wx(1)*wy(kount)),(wy(kount)+wy(1))*abs(wx(kount)*wy(1)-wx(1)*wy(kount))];
    tukmed=(tukmed/(3*som))+[(xsum/kount),(ysum/kount)];
else
    tukmed=[xsum,ysum];
end