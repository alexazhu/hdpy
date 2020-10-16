from Data import Data_Simulate as simu

# Generate Data
testsize = 50
n = 100
p = 500
s0 = 3
b = 1
snr = 1
sigma = "Exp_decay"
random_coefs = False

head = sigma + str(n) + '_' + str(p) + '_s' \
       + str(s0) + '_b' + str(b)

i =1

while i <= testsize:
    print(i)
    SIMU = simu.simulation(n=n, p=p, snr=snr, type=sigma, seed=i)

    beta,active = SIMU.get_coefs(s0=s0, random_coef=False, b=b,
                                 unif_up=None, unif_lw=None,
                                 save=True,path="./Exp_decay/"+head+"_betas/", filename=str(i))

    X, Y = SIMU.get_data(verbose=False, save=True, path="./Exp_decay/"+head+"_XY/", filename=str(i))
    i = i+1

# Fitting
