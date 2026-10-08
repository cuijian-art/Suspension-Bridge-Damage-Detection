function [ Entropystruct ] = EntropyFeatures(data)

%********************************计算熵特征值*****************************
    dim = 2;
    r = 0.2*std(data);
    tau = 1;
    sampEn = SampleEntropy( dim, r, data, tau );
    Entropystruct.sample=sampEn; % 样本熵
    
    pe = PermutationEntropy(data,dim,tau);
    Entropystruct.pe=pe; % 排列熵

    n = 2;
    FuzEn = FuzzyEntropy(data,dim,r,n,tau); 
    Entropystruct.fuzzy=FuzEn;   % 模糊熵 

    ApEn = ApproximateEntropy( dim, r, data, tau );
    Entropystruct.appro=ApEn;   % 近似熵 

    window_size = length(data)/3;
    energy = EnergyEntropy(data, window_size);
    Entropystruct.energy=energy;   % 能量熵 

    infor = information_entropy(data);
    Entropystruct.infor=infor;   % 信息熵

end