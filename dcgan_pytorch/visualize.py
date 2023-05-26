import numpy as np
import matplotlib.pyplot as plt

for i in range(5):
    data = np.load(f".\\data\\d_{i}.npz")

    delta_vmap = data['delta_vmap']
    dmap = data['dmap']
    nmap = data['nmap']

    c = plt.imshow(nmap)
    plt.show()

    c = plt.imshow(dmap, cmap='Greens', vmin=np.min(np.min(dmap)), vmax=np.max(np.max(dmap)),
                   interpolation='nearest', origin='lower')
    plt.colorbar(c)

    plt.title('map', fontweight="bold")
    plt.show()

    print(f"d_{i}")
    print(f'Max Val. = {np.max(np.max(dmap))}')
    print(f'Min Val. = {np.min(np.min(dmap))}')
