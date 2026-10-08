function [ waveletstruct ] =  waveletFeatures( src)
%计算小波域统计特征
%wpcoef：找小波包分解的系数，N层小波包分解后将信号分成2^N个子带，没个子带的信号长度是原是信号的1/2^N倍，其实就是将原始信号化成2^N段，每段的长度是相等的且比原信号短
%wprcoef：是小波包分解系数重构，将分解的各个子带的信号拓展为原始信号那么长，
%两者的区别就是wprcoef是wpcoef的拓展（长度与原始信号一样），他们每个节点的能量是一样的

%%小波熵是一个统称，不是一种具体的熵算法。
%旗下包括小波时间熵、小波尺度熵(小波特征尺度熵)、小波时频熵、小波能量熵(小波能谱熵)
% 小波相关尺度熵(小波相关特征尺度熵)、小波奇异熵、小波方差谱熵、小波空间特征熵(小波空间特征尺度熵)


n=3;%3层小波包分解
T=wpdec(src,n,'db6'); %3层小波包分解   得到8个子带
for i=1:2^n   %%重构分量，使信号和原始信号一样长
       E(i)=norm(wprcoef(T,[n,i-1]),2)*norm(wprcoef(T,[n,i-1]),2); %求第i个节点的二范数，乘以平方才是能量
end

E_total=sum(E);%总能量

%8个子带的小波能量比
waveletstruct.p1=E(1)/E_total; 
waveletstruct.p2=E(2)/E_total; 
waveletstruct.p3=E(3)/E_total; 
waveletstruct.p4=E(4)/E_total; 
waveletstruct.p5=E(5)/E_total; 
waveletstruct.p6=E(6)/E_total; 
waveletstruct.p7=E(7)/E_total; 
waveletstruct.p8=E(8)/E_total; 


%绘制条形图
% p=[waveletstruct.p1,waveletstruct.p2,waveletstruct.p3,waveletstruct.p4,...
%     waveletstruct.p5,waveletstruct.p6,waveletstruct.p7,waveletstruct.p8];
% bar(p);
% set(gca,'XTickLabel',{'(3,0)','(3,1)','(3,2)','(3,3)' ,'(3,4)','(3,5)','(3,6)','(3,7)'});

%小波能量熵(小波能谱熵)
waveletstruct.energyE=-sum(E/E_total.*log(E/E_total)); 


%8个子带的小波熵(小波尺度熵)   在各尺度上求的是shannon熵=小波尺度熵(小波特征尺度熵)
waveletstruct.E1=wentropy(wpcoef(T,[3,0]),'shannon');%
waveletstruct.E2=wentropy(wpcoef(T,[3,1]),'shannon');%
waveletstruct.E3=wentropy(wpcoef(T,[3,2]),'shannon');%
waveletstruct.E4=wentropy(wpcoef(T,[3,3]),'shannon');%
waveletstruct.E5=wentropy(wpcoef(T,[3,4]),'shannon');%
waveletstruct.E6=wentropy(wpcoef(T,[3,5]),'shannon');%
waveletstruct.E7=wentropy(wpcoef(T,[3,6]),'shannon');%
waveletstruct.E8=wentropy(wpcoef(T,[3,7]),'shannon');%


%小波奇异谱熵
%%先求取奇异值，将奇异值构造向量即得到奇异值谱，然后计算熵值即可
%第一步：小波分解，得到各分量信号，对系数重构，得到完整系数
for i=1:2^n   %%重构分量，使信号和原始信号一样长
       q(:,i)=wprcoef(T,[n,i-1]); %求系数
end

%奇异值分解
s=svd(q);
S=sum(s);
dim=length(s);
for i=1:dim
    p(i)=s(i)/S;
end
waveletstruct.qyshang=-sum(p.*log(p));%小波奇异熵




end


