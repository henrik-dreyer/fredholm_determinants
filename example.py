##########################################
#######   IMPORTS                  #######
##########################################

from coherent_state import CoherentState
from xy_model import K, dfdt
from matplotlib import pyplot as plt




###########################################
#######   SET UP TRAJECTORY         #######
#######   IN PARAMETER SPACE        #######
#######   FOR MORE COMPLEX          #######
#######   TRAJECTORIES, SET         #######
#######   DERIVATIVES               #######
###########################################

def dgammadt(gamma):
    out = 0             #gamma = gamma_initial (const)
    return out

def dhdt(h):
    speedh = 1
    out = speedh        #h = h_initial + speedh * t
    return out


##########################################
#######   INITIALIZATION           #######
##########################################

q = 10  # Discretization points. Q = 2^q = 1024
        # Controls error of Fredholm determinant as 1/Q^2
r = 1   # Effective system size R = (2*r+1)/2 * Q = 1536
h = 0   # XY transverse field
gamma=1 # XY anisotropy


f_initial= lambda k: 1j*K(0, 1, h, gamma, k)    #Set f(t=0, k). Imported from xy_model
f_params = dict(q=q, r=r, f_initial=f_initial)  #Collect parameters and
f = CoherentState(f_params)                     #create an instance of the Coherent State class



##########################################
#######   TEST INITIAL POINT       #######
##########################################
m=f.compute_m()
print(r'X-Magnetization in XY-model for h={} and gamma={}: {}'.format(h,gamma,m))



##########################################
#######   EVOLVE BY LINEAR STEPPER #######
##########################################
T_final = 1
n_steps = 100
dt = T_final/n_steps
t=0
dfdt_lambda = lambda f,k: dfdt(f,k,gamma,h)

ts = []
ms = []
for j in range(n_steps):
    h = h+dt*dhdt(h)
    gamma = gamma + dt*dgammadt(gamma)
    t=t+dt
    f.evolve(dfdt_lambda, dt)
    m = f.compute_m()

    txt_file = "t={:.3f}.txt".format(t) #Write to text file
    f.write_to_txt(txt_file)

    #Add data and plot
    ms.append(m)
    ts.append(t)
    plt.plot(ts, ms, 'o')
    plt.xlabel('$t$')
    plt.ylabel('$<m_X(t)>$')
    plt.title('Time evolution in XY model')
    plt.show()