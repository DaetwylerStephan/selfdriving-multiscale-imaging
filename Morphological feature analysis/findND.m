function varargout=findND(X,varargin)
%Find non-zero elements in ND-arrays. Replicates all behavior from find.

%https://www.mathworks.com/matlabcentral/fileexchange/64383-findnd
% The syntax is equivalent to the built-in find, but extended to multi-dimensional input.
%
% [...] = findND(X,K) returns at most the first K indices. K must be a positive scalar of any type.
%
% [...] = findND(X,K,side) returns either the first K or the last K inidices. The input side  must
% be a char, either 'first' or 'last'. The default behavior is 'first'.
%
% [I1,I2,I3,...,In] = findND(X,...) returns indices along all the dimensions of X.
%
% [I1,I2,I3,...,In,V] = findND(X,...) returns indices along all the dimensions of X, and
% additionally returns a vector containg the values.
%
% Note for Matlab 6.5:
% The syntax with more than one input is present in the online doc for R14 (Matlab 7.0), so this
% might be the latest release without support for this syntax.
%
%  _______________________________________________________________________
% | Compatibility | Windows 10  | Ubuntu 20.04 LTS | MacOS 10.15 Catalina |
% |---------------|-------------|------------------|----------------------|
% | ML R2020a     |  works      |  not tested      |  not tested          |
% | ML R2018a     |  works      |  works           |  not tested          |
% | ML R2015a     |  works      |  works           |  not tested          |
% | ML R2011a     |  works      |  works           |  not tested          |
% | ML 6.5 (R13)  |  works      |  not tested      |  not tested          |
% | Octave 5.2.0  |  works      |  works           |  not tested          |
% | Octave 4.4.1  |  works      |  not tested      |  works               |
% """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
%
% Version: 1.2.1
% Date:    2020-07-06
% Author:  H.J. Wisselink
% Licence: CC by-nc-sa 4.0 ( http://creativecommons.org/licenses/by-nc-sa/4.0 )
% Email=  'h_j_wisselink*alumnus_utwente_nl';
% Real_email = regexprep(Email,{'*','_'},{'@','.'})
%Parse inputs
if ~(isnumeric(X) || islogical(X)) || numel(X)==0
    error('HJW:findND:FirstInput',...
        'Expected first input (X) to be a non-empty numeric or logical array.')
end
switch nargin
    case 1 %[...] = findND(X);
        side='first';
        K=inf;
    case 2 %[...] = findND(X,K);
        side='first';
        K=varargin{1};
        if ~(isnumeric(K) || islogical(K)) || numel(K)~=1 || any(K<0)
            error('HJW:findND:SecondInput',...
                'Expected second input (K) to be a positive numeric or logical scalar.')
        end
    case 3 %[...] = FIND(X,K,'first');
        K=varargin{1};
        if ~(isnumeric(K) || islogical(K)) || numel(K)~=1 || any(K<0)
            error('HJW:findND:SecondInput',...
                'Expected second input (K) to be a positive numeric or logical scalar.')
        end
        side=varargin{2};
        if ~isa(side,'char') || ~( strcmpi(side,'first') || strcmpi(side,'last'))
            error('HJW:findND:ThirdInput','Third input must be either ''first'' or ''last''.')
        end
        side=lower(side);
    otherwise
        error('HJW:findND:InputNumber','Incorrect number of inputs.')
end
%parse outputs
nDims=length(size(X));
%allowed outputs: 0, 1, nDims, nDims+1
if nargout>1 && nargout<nDims
    error('HJW:findND:Output','Incorrect number of output arguments.')
end
persistent OldSyntax
if isempty(OldSyntax)
    OldSyntax=ifversion('<',7,'Octave','<',3);
end
varargout=cell(nargout,1);
if OldSyntax
    %The find(X,k,side) syntax was introduced between 6.5 and 7
    if nargout>nDims
        [ind,col_index_equal_to_one,val]=find(X(:));%#ok no tilde pre-R2009b
        %X(:) converts X to a column vector. Treating X(:) as a matrix
        %forces val to be the actual value, instead of the column index.
        if length(ind)>K
            if strcmp(side,'first') %select first K outputs
                ind=ind(1:K);
                val=val(1:K);
            else                    %select last K outputs
                ind=ind((end-K+1):end);
                val=val((end-K+1):end);
            end
        end
        [varargout{1:(end-1)}] = ind2sub(size(X),ind);
        varargout{end}=val;
    else
        ind=find(X);
        if length(ind)>K
            if strcmp(side,'first')
                %select first K outputs
                ind=ind(1:K);
            else
                %select last K outputs
                ind=ind((end-K):end);
            end
        end
        [varargout{:}] = ind2sub(size(X),ind);
    end
else
    if nargout>nDims
        [ind,col_index_equal_to_one,val]=find(X(:),K,side);%#ok<ASGLU>
        %X(:) converts X to a column vector. Treating X(:) as a matrix forces val to be the actual
        %value, instead of the column index.
        [varargout{1:(end-1)}] = ind2sub(size(X),ind);
        varargout{end}=val;
    else
        ind=find(X,K,side);
        [varargout{:}] = ind2sub(size(X),ind);
    end
end
end
function tf=ifversion(test,Rxxxxab,Oct_flag,Oct_test,Oct_ver)
%Determine if the current version satisfies a version restriction
%
% To keep the function fast, no input checking is done. This function returns a NaN if a release
% name is used that is not in the dictionary.
%
% Syntax:
% tf=ifversion(test,Rxxxxab)
% tf=ifversion(test,Rxxxxab,'Octave',test_for_Octave,v_Octave)
%
% Output:
% tf       - If the current version satisfies the test this returns true.
%            This works similar to verLessThan.
%
% Inputs:
% Rxxxxab - Char array containing a release description (e.g. 'R13', 'R14SP2' or 'R2019a') or the
%           numeric version.
% test    - Char array containing a logical test. The interpretation of this is equivalent to
%           eval([current test Rxxxxab]). For examples, see below.
%
% Examples:
% ifversion('>=','R2009a') returns true when run on R2009a or later
% ifversion('<','R2016a') returns true when run on R2015b or older
% ifversion('==','R2018a') returns true only when run on R2018a
% ifversion('==',9.8) returns true only when run on R2020a
% ifversion('<',0,'Octave','>',0) returns true only on Octave
%
% The conversion is based on a manual list and therefore needs to be updated manually, so it might
% not be complete. Although it should be possible to load the list from Wikipedia, this is not
% implemented.
%
%  _______________________________________________________________________
% | Compatibility | Windows 10  | Ubuntu 20.04 LTS | MacOS 10.15 Catalina |
% |---------------|-------------|------------------|----------------------|
% | ML R2020a     |  works      |  not tested      |  not tested          |
% | ML R2018a     |  works      |  works           |  not tested          |
% | ML R2015a     |  works      |  works           |  not tested          |
% | ML R2011a     |  works      |  works           |  not tested          |
% | ML 6.5 (R13)  |  works      |  not tested      |  not tested          |
% | Octave 5.2.0  |  works      |  works           |  not tested          |
% | Octave 4.4.1  |  works      |  not tested      |  works               |
% """""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
%
% Version: 1.0.2
% Date:    2020-05-20
% Author:  H.J. Wisselink
% Licence: CC by-nc-sa 4.0 ( creativecommons.org/licenses/by-nc-sa/4.0 )
% Email=  'h_j_wisselink*alumnus_utwente_nl';
% Real_email = regexprep(Email,{'*','_'},{'@','.'})
%The decimal of the version numbers are padded with a 0 to make sure v7.10 is larger than v7.9.
%This does mean that any numeric version input needs to be adapted. multiply by 100 and round to
%remove the potential for float rounding errors.
%Store in persistent for fast recall (don't use getpref, as that is slower than generating the
%variables and makes updating this function harder).
persistent  v_num v_dict octave
if isempty(v_num)
    %test if Octave is used instead of Matlab
    octave=exist('OCTAVE_VERSION', 'builtin');
    
    %get current version number
    v_num=version;
    ii=strfind(v_num,'.');
    if numel(ii)~=1,v_num(ii(2):end)='';ii=ii(1);end
    v_num=[str2double(v_num(1:(ii-1))) str2double(v_num((ii+1):end))];
    v_num=v_num(1)+v_num(2)/100;
    v_num=round(100*v_num);%remove float rounding errors
    
    %get dictionary to use for ismember
    v_dict={...
        'R13' 605;'R13SP1' 605;'R13SP2' 605;'R14' 700;'R14SP1' 700;'R14SP2' 700;'R14SP3' 701;...
        'R2006a' 702;'R2006b' 703;'R2007a' 704;'R2007b' 705;'R2008a' 706;'R2008b' 707;...
        'R2009a' 708;'R2009b' 709;'R2010a' 710;'R2010b' 711;'R2011a' 712;'R2011b' 713;...
        'R2012a' 714;'R2012b' 800;'R2013a' 801;'R2013b' 802;'R2014a' 803;'R2014b' 804;...
        'R2015a' 805;'R2015b' 806;'R2016a' 900;'R2016b' 901;'R2017a' 902;'R2017b' 903;...
        'R2018a' 904;'R2018b' 905;'R2019a' 906;'R2019b' 907;'R2020a' 908};
end
if octave
    if nargin==2
        warning('HJW:ifversion:NoOctaveTest',...
            ['No version test for Octave was provided.',char(10),...
            'This function might return an unexpected outcome.']) %#ok<CHARTEN>
        %Use the same test as for Matlab, which will probably fail.
        L=ismember(v_dict(:,1),Rxxxxab);
        if sum(L)~=1
            warning('HJW:ifversion:NotInDict',...
                'The requested version is not in the hard-coded list.')
            tf=NaN;return
        else
            v=v_dict{L,2};
        end
    elseif nargin==4
        %undocumented shorthand syntax: skip the 'Octave' argument
        [test,v]=deal(Oct_flag,Oct_test);
        %convert 4.1 to 401
        v=0.1*v+0.9*fix(v);v=round(100*v);
    else
        [test,v]=deal(Oct_test,Oct_ver);
        %convert 4.1 to 401
        v=0.1*v+0.9*fix(v);v=round(100*v);
    end
else
    %convert R notation to numeric and convert 9.1 to 901
    if isnumeric(Rxxxxab)
        v=0.1*Rxxxxab+0.9*fix(Rxxxxab);v=round(100*v);
    else
        L=ismember(v_dict(:,1),Rxxxxab);
        if sum(L)~=1
            warning('HJW:ifversion:NotInDict',...
                'The requested version is not in the hard-coded list.')
            tf=NaN;return
        else
            v=v_dict{L,2};
        end
    end
end
switch test
    case '=='
        tf= v_num == v;
    case '<'
        tf= v_num <  v;
    case '<='
        tf= v_num <= v;
    case '>'
        tf= v_num >  v;
    case '>='
        tf= v_num >= v;
end
end