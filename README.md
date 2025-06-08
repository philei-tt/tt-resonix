# tt-resonix

## Goal
tt-resonix is a pet-project. It's purpose is to implement finite-difference wave propagation
simulation on tenstorrent hardware and compare performance to cpu and gpu

## Performance
[Performance is tracked here](./docs/performance_acoustic_2d.md)

## Implementations:

- Numpy baseline | 2d acoustic | no staggering | Neumann boundary condition: [here](./fdtd/python/fdtd2d.py)

## Installation
### Requirements:
- python3.9+

### Setup
```bash
pip3 install -r requirements.txt
```

### Run:
**- Numpy baseline**
```bash
# Run simulation
python3 ./fdtd/python/fdtd2d.py ./gt/acoustic_1st_ord_2d/0/config.json --tqdm --output wavefield.npz
# Vizualize simulation
python3 ./tools/viz_sim.py --clip=0.999 wavefield.npz
```

### Tools
**- Compare with baseline**