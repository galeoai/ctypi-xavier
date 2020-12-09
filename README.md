# Ctypi-Cuda
Implementation of ctypi algorithm using CUDA on Xavier

## Description 

## Dependencies
- opencv
- CUDA
- hdf5


## notes

### Segmentation fault (core dumped)

```bash
ulimit -s unlimited
```

### speedup Xavier 
```bash
sudo /usr/sbin/nvpmodel -m 0 
sudo /usr/bin/jetson_clocks
```
