
a=3.86;                                     % lattice parameter for Bi2212        
patch_dim = 128;                            % dimension of patches used for training
m = patch_dim^2;
ksr = 300;                                  % # of pixels in momentum axis
wsr = 400;                                  % # of pixels in energy axis
evec = linspace(-0.2,0.1,wsr);              % energy vector
dw = evec(2)-evec(1);
fbin = find(evec==min(abs(evec)));          % location of fermi energy
w = reshape(evec,1,1,wsr); w = repmat(w,ksr,ksr,1);  %Augment dimensionality for ease of calculations later on
kx_ = linspace(-1/(2*a),1/(2*a),ksr);       % momentum scale (1/A)
[kx,ky] = meshgrid(kx_,kx_); 


kB = 8.61733034*10^-5;         % Boltzmann constant (ev/k)


for ind = 1

    T = 80*rand + 20;                   % random temps in Kelvin
    kt = kB*T;                          % handy energy scale
    del = 0.032 + 0.02*rand - 0.01;     % random variation of antinodal gap
    gamma = 0.04 + 0.02*rand-0.01;      % random variation of broadening

    % Varied band dispersions
    c0 =  0.5 + 0.1*rand-0.05;
    c1 =  0.1305 + 0.04*rand-0.02;    
    c2 = -0.5951 + 0.1*rand-0.05;
    c3 =  0.1636 + 0.04*rand-0.02;  
    c4 = -0.0519 + 0.02*rand-0.01;  
    c5 = -0.1117 + 0.04*rand-0.02;  
    c6 =  0.0510 + 0.02*rand-0.01;   
    % Tight binding fit
    ek = c0*(c1 + (c2/2)*(cos(2*pi*kx*a) + cos(2*pi*ky*a)) + c3*(cos(2*pi*kx*a).*cos(2*pi*ky*a))...
        + (c4/2)*(cos(2*pi*2*kx*a) + cos(2*pi*2*ky*a)) + (c5/2)*(cos(2*pi*2*kx*a).*cos(2*pi*ky*a)+...
        cos(2*pi*kx*a).*cos(2*pi*2*ky*a)) + c6*(cos(2*pi*2*kx*a).*cos(2*pi*2*ky*a)));    %  dispersion (eV)

    gapk = (del/2)*abs((cos(2*pi*kx*a)-cos(2*pi*ky*a)));  % d-wave gap
    gapk = repmat(gapk,1,1,wsr);            % Augment dimensionality for ease of calculations later on
    om = 1.3*del;                           % approximate momentum dependence of mode energy
    om_gap = om+gapk;                       % dip energy

    %Real part of self-energy
    reself = (gamma/pi)*log(abs((w-om_gap)./(w+om_gap)));

    %Imaginary part of self-energy
    ind_array = om_gap < abs(w);
    imself = gamma*(double(ind_array));

    %Self-energy and renormalization function Z.
    self = reself+1i*imself;
    z = 1 - self./w;

    %Add a coherence factor to the energies (d). To fit the SC data with a QP
    %peak and an incoherent feature.
    d = -0.015;
    A0 = (1/pi)*imag((z.*(w+1i*d)+ek)./(z.^2.*((w+1i*d).^2-gapk.^2)-ek.^2));

    % Convolve EDC with 10-15 meV resolution function (Note that the Gaussian must be
    % centered at middle of energy interval of interest)   
    w0 = (evec(end)+evec(1))/2;               % Gaussian center
    sig1 = 0.007+0.005*rand;                  % Gaussian width (eV)
    r1 = (1/(sig1*sqrt(2*pi))).*exp(-0.5*(evec-w0).^2/sig1^2);    % Gaussian resolution function 
    
    %In MATLAB, we can easily vectorize the convolution, which makes it
    %fast.
    A1 = dw*convn(permute(A0,[3,1,2]),r1','same');      % broadened spectral function (label)
    A1 = permute(A1,[2,3,1]);
  

    %%%%%%%%   Now create the sample %%%%%%%%%%

    % A1 will be taken to be the the (broad) label. To create the sample, I
    % convolve A1 with a thin Gaussian and apply Poisson noise.

    % Convolve EDC with additional 1 meV resolution function 
    w0 = (evec(end)+evec(1))/2;             % Gaussian center
    sig2 = 0.001;                           % additional broadening for the sample         
    r2 = (1/(sig2*sqrt(2*pi))).*exp(-0.5*(evec-w0).^2/sig2^2);    % Gaussian resolution function 
    A2 = dw*convn(permute(A1,[3,1,2]),r2','same');      % new spectral function (sample)
    A2 = permute(A2,[2,3,1]);

    % Multiply by Fermi functions
    width1 = sqrt(kt^2+sig1^2)/4.2;
    f1 = 1./(exp(w/width1)+1);
    A1 = A1.*f1;
    width2 = sqrt(kt^2+sig1^2+sig2^2)/4.2;
    f2 = 1./(exp(w/width2)+1);
    A2 = A2.*f2;

    %Remove edges affected by convolutions
    cutoff = 50 ; 
    A0 = A0(:,:,cutoff:end-cutoff);
    A1 = A1(:,:,cutoff:end-cutoff);
    A2 = A2(:,:,cutoff:end-cutoff);  
    om_gap = om_gap(:,:,cutoff:end-cutoff);
    w1 = w(:,:,cutoff:end-cutoff);   
    evec1 = evec(cutoff:end-cutoff);
    fbin = find(evec1==min(abs(evec1)));     %location of fermi energy
    wsr1 = length(evec1);

    % For cuts in the M-Y direction (as we usually measure), do the following:

    scale = 0.8+0.4*rand-0.2;     % scale factor controlling noise

    while 1 
        k =  floor(ksr*scale);           % random ky momentum  

        disp0 = A1(:,k,:);     disp0 = reshape(disp0,ksr,wsr1);      % label dispersion
        disp = A2(:,k,:);       disp = reshape(disp,ksr,wsr1);       % broadened dispersion

        % Randomly select mini-batch
        mbx = 1+floor((ksr-patch_dim)*rand);               % patch starting x-coord
        mby = 1+floor((wsr1-patch_dim)*rand);              % patch starting y-coord

        x = disp0(mbx:mbx+patch_dim-1,mby:mby+patch_dim-1);     % label patch     
        x = x/max(max(x));                          % x is now the "ground truth"

        % Check that label is "good" (i.e., not just a bunch of zeros)
        vol = sum(sum(x))/m;     
        if vol>0.1
           break
        end
    end

    y = disp(mbx:mbx+patch_dim-1,mby:mby+patch_dim-1);    
    y = y/max(max(y));                   

    % Apply Poisson noise
    scale = 1000+2000*rand;          %  max intensities between 1000 and 3000
    y = round(scale*y/max(max(y))); 
    y = uint16(y);
    y = imnoise(y,'poisson');
    y = double(y);  y = y/scale;     % broadened,noisy, normalized patch; this is run through network

    % User should then save the generated images x (label) and y (sample) and keep looping
end

imagesc(x)
figure;imagesc(y)

