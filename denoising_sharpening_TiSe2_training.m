% 1T-TiSe2  bands for training
% Seeing as I have failed to process the dichalcogenide bands given to me,
% I will try to create bands more representative of the target ones, to
% train the network more properly.

% Use tight binding fits by Norman for the Bi2212 dispersion. I create
% n_disp different 3-D dispersions and then broaden them and add some
% Gaussian noise

a = 3.86;                         % lattice spacing (Angstrom)
ksr = 300;                        % momentum sampling rate
wsr = 400;                        % energy sampling rate
evec = linspace(-0.5,0.1,wsr);
fbin = find(evec==min(abs(evec)));     %location of fermi energy
w = reshape(evec,1,1,wsr); w = repmat(w,ksr,ksr,1);  %Augment dimensionality for ease of calculations later on
dw = evec(2)-evec(1);
kx_ = linspace(-1/(2*a),1/(2*a),ksr);     % momentum scale (1/A)

[kx,ky] = meshgrid(kx_,kx_); 

tic
      
patch_dim = 128;      
f1 = 9;                 % filter dimension layer1
num1 = 64;              % number of filters layer 1
f2 = 1;                 % filter dimension layer 2
num2 = 32;              % # of filters layer 2
f3 = 5;             	% filter dimension layer 3
num3 = 1;               % # of filters layer 3

trim = (f1+f2+f3-3)/2; 
m = patch_dim-2*trim ;            % size of output after all convolutions

% Tight binding parameters proposed by Norman (Physical Review B, 75, 184514)
% (eV)
c1 =  0.1305;     
c2 = -0.5951;
c3 =  0.1636;
c4 = -0.0519;
c5 = -0.1117;
c6 =  0.0510;

cutoff = 50;             % pixels to remove after convolution (both sides)
off = 0.18;              % band offset (eV)
gamma0 = 0.007;           % label peak width (eV)

for ind = 1
    
    kT = 0.01 + 0.01*rand-0.005;
    
    %Bandwidth prefactor:
    c0 = 0.3+0.05*rand;

    %Main cuprate-like band:
    ek =  c0*(c1 + (c2/2)*(cos(2*pi*kx*a) + cos(2*pi*ky*a)) + c3*(cos(2*pi*kx*a).*cos(2*pi*ky*a))...
        + (c4/2)*(cos(2*pi*2*kx*a) + cos(2*pi*2*ky*a)) + (c5/2)*(cos(2*pi*2*kx*a).*cos(2*pi*ky*a) + ...
        cos(2*pi*kx*a).*cos(2*pi*2*ky*a)) + c6*(cos(2*pi*2*kx*a).*cos(2*pi*2*ky*a)));    % (eV)

    % Weakly dispersing contribution:
    c7 = 0.01 + 0.01*rand-0.005;
    c8 = 0.004 + 0.004*rand-0.002;
    ek1 = c7+(c7/2)*(cos(2*pi*kx*a) + cos(2*pi*ky*a))-(c8/2)*(cos(2*pi*2*kx*a)+cos(2*pi*2*ky*a)) - c8*cos(2*pi*kx*a).*cos(2*pi*ky*a);

    % Gap function (eV):
    del = 0.05 + 0.06*rand-0.02;
    gapk = (del/2)*abs((cos(2*pi*kx*a)-cos(2*pi*ky*a)));
    %Augment dimensionality for ease of calculations later on
    gapk = repmat(gapk,1,1,wsr);   

    % Bogoliubov dispersion (eV) and coherence factors:
    Ek1 = sqrt(ek.^2+2*gapk.^2);
    vk2 = 0.5*(1-ek./Ek1);
    uk2 = 1-vk2;

    % Weights of different branches
    wei1 = 1 + rand - 0.5;      % strength of bottom wings  
    wei2 = 1 + rand - 0.5;      % strength of bottom hump   
    wei3 = 1 + rand - 0.5;      % strength of top wings
    wei4 = 1 + rand - 0.5;      % strength of top hump   
    

    A1 = (wei1*uk2/pi)* gamma0./((w+Ek1+off).^2+gamma0^2);
    A2 = (wei2*vk2/pi)* gamma0./((w+Ek1+off).^2+gamma0^2);
    A3 = (wei3*uk2/pi)* gamma0./((w-Ek1+off).^2+gamma0^2);
    A4 = (wei4*vk2/pi)* gamma0./((w-Ek1+off).^2+gamma0^2);

    % Total "label" spectral function:
    A0 = A1 + A2 + A3 + A4;

    % Now, we "corrupt" these labels to generate the samples 

    % Convolve EDC with 20-40 meV resolution function (Note that the Gaussian must be
    % centered at middle of energy interval of interest)   
    w0 = (evec(end)+evec(1))/2;     
    sig = 0.02 + 0.02*rand;  
    % Gaussian resolution function:
    r = (1/(sig*sqrt(2))).*exp(-0.5*(evec-w0).^2/sig^2); 
    % Broadened spectral function:
    A = dw*convn(permute(A0,[3,1,2]),r','same');      
    A = permute(A,[2,3,1]);

    % Multiply by  fermi functions
    width0 = kT/4.2;
    f0 = 1./(exp(w/width0)+1);
    width1 = sqrt(sig^2+kT^2)/4.2;
    f = 1./(exp(w/width1)+1);
    A0 = A0.*f0;
    A = A.*f0;
    
    % Remove edges affected by convolution
    A = A(:,:,cutoff:end-cutoff);  
    A0 = A0(:,:,cutoff:end-cutoff);  
    w1 = w(:,:,cutoff:end-cutoff);   
    evec1 = evec(cutoff:end-cutoff);
    fbin1 = find(evec==min(abs(evec1)));     %location of fermi energy
    wsr1 = length(evec1);
    f0 = f0(:,:,cutoff:end-cutoff);
    f = f(:,:,cutoff:end-cutoff);

    % Select random ky slice for label and sample:
    scale = 0.8+0.4*rand-0.2;
    k =  round(ksr*scale);            

    disp0 = A0(:,k,:);     disp0 = reshape(disp0,ksr,wsr1);     % label dispersion
    disp = A(:,k,:);       disp = reshape(disp,ksr,wsr1);       % broadened dispersion

    % Randomly crop patches from selected slices:
    mbx = 1+floor((ksr-patch_dim)*rand);               % patch starting x-coord
    mby = 1+floor((wsr1-patch_dim)*rand);              % patch starting y-coord
    x = disp0(mbx:mbx+patch_dim-1,mby:mby+patch_dim-1);     % label patch   

    % "Ground truth":
    x = x/max(max(x));                          

    % Function for modulating SAMPLE intensity at different momenta,
    % to simulate matrix element effects:
    k_red = kx_(mbx:mbx+patch_dim-1);
    lim1 =  randsample(20:patch_dim-20,1);    lim2 = randsample(1:lim1-10,1);
    k_min = k_red(lim2);  k_max = k_red(lim1);
    width1 = 0.01 + 0.005*rand-0.002;
    width2 = 0.01 + 0.005*rand-0.002;
    offset = 0.01*rand;
    fk = offset + 1./(exp((k_red-k_max)/width1)+1)+(1-1./(exp((k_red-k_min)/width2)+1));

    y = disp(mbx:mbx+patch_dim-1,mby:mby+patch_dim-1);   
    y = y.*fk';
    y = y/max(max(y)); 

    % Apply Poisson noise
    % The following "scale" should be adjusted to handle data collected with
    % different maximum intensities.
    scale = 1000+2000*rand;
    y = round(scale*y/max(max(y))); 
    y = uint16(y);
    y = imnoise(y,'poisson');
    y = double(y);                         

    % Final noisy, broadened, anisotropic "sample":
    y = y/scale;                     % broadened,noisy, normalized patch; this is run through network

    % NOTE: x and y are now normalized to take values between 0 and 1, but y has a
    % Poisson noise level representative of data with max. intensities around the
    % value "scale".

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % User should save x (label) and y (sample) images and keep looping
end

imagesc(x)
figure;imagesc(y)
