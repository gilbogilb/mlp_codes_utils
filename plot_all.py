#make the ID of a potential - plot all the info we have
#add check of if files exist and skip otherwise

import matplotlib.pyplot as plt
import numpy as np


#parity plots
e_parity = np.loadtxt('e-parity.dat')
f_parity = np.loadtxt('f-parity.dat')
s_parity = np.loadtxt('s-parity.dat')

parity_data   = [e_parity, f_parity, s_parity]
parity_names  = ['Energy per atom', 'Forces', 'Stress']
parity_labels = ['E/at [eV]', 'F (eV/A)', 'sigma [eV/A^3]']
parity_colors = ['r', 'g', 'b']

fig, axs = plt.subplots(1,3)

for ax, data, name, label in zip(axs, parity_data, parity_names, parity_labels):
    ax.plot(data[:,0], data[:,0],'k--')
    ax.plot(data[:,0], data[:,1],'o')
    ax.set_xlabel('reference '+label)
    ax.set_ylabel('model '+label)
    ax.set_title(name)

plt.suptitle('Parity plots')

#detached atoms curves
fig, axs = plt.subplots(2,1)
dimer = np.loadtxt('dimer_curve.dat')
adso  = np.loadtxt('adsorbate_curve.dat')

axs[0].plot(dimer[:,0], dimer[:,1])
axs[0].set_xlabel('dimer separation [A]')
axs[0].set_ylabel('energy [eV]')

axs[1].plot(adso[:,0], adso[:,1])
axs[1].set_xlabel('atom-surface separation [A]')
axs[1].set_ylabel('energy [eV]')

plt.suptitle('distanced atoms curves')

#cluster excess energy curves
ico_data = np.loadtxt('ico-exc.dat')
deca_data = np.loadtxt('deca-exc.dat')
octa_data = np.loadtxt('octa-exc.dat')

plt.figure()
plt.scatter(ico_data[:,0], ico_data[:,1], label='ico')
plt.scatter(deca_data[:,0], deca_data[:,1], label='deca')
plt.scatter(octa_data[:,0], octa_data[:,1], label='octa')

plt.xlabel('N')
plt.ylabel('Exc. energy [eV]')
plt.xscale('log')
plt.legend()

plt.show()
    


