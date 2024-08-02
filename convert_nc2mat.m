% function convert_nc2mat(year, doy, fdir)
clc;
clear;

% self defined
year = 2024;
doy = 146;
fdir = '.';    % path to TEC .nc file
% 

if ispc()
    delimiter = '\';
else 
    if isunix()
        delimiter = '/';
    end
end

fname = ['BeiDou_TEC_', num2str(doy, '%03.0f'), '_01s_', num2str(year)];
fnc = [fdir, delimiter, fname, '.nc'];
svs = ["C01", "C02", "C03", "C04", "C05"];

%% 加载nc变量
% % for earlier matlab versions --------------------------------------------------
NCsta = h5read (fnc,'/observer'); 
NCsv = h5read (fnc,'/sv');
NCt = h5read (fnc,'/time');

NClon = h5read (fnc,'/lon');
NClat = h5read (fnc,'/lat');
NCele = h5read (fnc,'/elevation');    % units：degree

NCstec = h5read (fnc,'/sTEC');    % relativeTEC (slant)
NCdcbs = h5read (fnc,'/DCBs');    % 用来计算绝对TEC，absoluteTEC = (relativeTEC - DCBs)*sin(elevation)

% fprintf('nc variables is loaded !\n'); 

% % for later matlab versions ----------------------------------------------------
% NCsta = ncread (fnc,'observer'); 
% NCsv = ncread (fnc,'sv');
% NCt = ncread (fnc,'time');
% 
% NClon = ncread (fnc,'lon');
% NClat = ncread (fnc,'lat');
% NCele = ncread (fnc,'elevation');
% 
% NCstec = ncread (fnc,'sTEC');
% NCdcbs = ncread (fnc,'DCBs');    % 用来计算绝对TEC，absoluteTEC = (relativeTEC - DCBs)*sin(elevation)

% % fprintf('nc variables is loaded !\n');

%% 主方案: 
% 1st.从nc数据中提取站点IPP信息

ipps = cell(size(NCsta,1),1);
for staNum=1:size(NCsta,1)
    ipps{staNum} = NaN(5,3);
    for prn = 1:5
        if ~ismember(svs(prn), NCsv)
            continue
        end
        idx = find(NCsv==svs(prn), 1);
        ipps{staNum}(prn, 1) = NClon(idx,staNum);
        ipps{staNum}(prn, 2) = NClat(idx,staNum);
        ipps{staNum}(prn, 3) = NCele(idx,staNum);
    end
end
clear NClon NClat ele;

fprintf('ipp information is extracted !\n');

%% 
% 2nd.存取tec变量
tec = cell(size(NCsta,1), 7);
for staNum=1:size(NCsta,1)
    tec{staNum,1}=NCsta{staNum};
    tec{staNum,2}=ipps{staNum};
    
    for prn = 1:5
        data = [];
        if ismember(svs(prn), NCsv)
            idx = find(NCsv==svs(prn), 1);
            dat = NCstec(:,idx,staNum);
            
            data(:,1) = NCt(~isnan(dat)); 
            data(:,2) = dat(~isnan(dat));  

            % --- 计算绝对TEC， 注释得到相对TEC ---------------
            dcbs = NCdcbs(idx, staNum);
            elevation = NCele(idx, staNum); 
            data(:,2) = (data(:,2) - dcbs) * sind(elevation);   
            % -----------------------------------------------
            tec{staNum, prn+2} = data;
        else
            tec{staNum, prn+2} = [];
        end
    end
end
clear NCsta NCsv NCele NCt NCstec ipps dat data;

save([fname,'.mat'], 'tec');
fprintf('converting file(%s) to mat is complete !\n', [fname, '.nc']);
% return 0

%% 备用方案: 直接读取提前存好的站点IPP信息并存入tec数据
% load('CstaNameNat170_2021.mat')
% tec = cell(size(CstaName,1), 7);
% for staNum=1:size(CstaName,1)
%     tec{staNum,1}=CstaName{staNum,1};
%     tec{staNum,2}=CstaName{staNum,2};
%     for shift=1:size(NCsta,1)
%         if strcmpi(CstaName{staNum,1},NCsta{shift})
%             for prn = 1:5
%                 data = [];
%                 if ismember(svs(prn), NCsv)
%                     idx = find(NCsv==svs(prn), 1);
%                     dat = NCstec(:,idx,shift);
%                     data(:,1) = NCt(~isnan(dat));
%                     data(:,2) = dat(~isnan(dat));
%                     % --- 计算绝对TEC， 注释得到相对TEC ---------------
%                     dcbs = NCdcbs(idx, staNum);
%                     elevation = NCele(idx, staNum); 
%                     data(:,2) = (data(:,2) - dcbs) * sind(elevation);   
%                     % -----------------------------------------------
%                     tec{staNum, prn+2} = data;
%                 else
%                     tec{staNum, prn+2} = [];
%                 end
%             end
%         end
% 
%     end
% end
% 
% % save([fname,'.mat'], 'tec');
% fprintf('converting file(%s) to mat is complete !\n', [fname, '.nc']);
% return 0

%%