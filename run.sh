#!/usr/bin/env bash
mpirun -np 4 pytriqs run_nonint_1orb.py
mpirun -np 4 pytriqs run_nonint_2orb.py
pytriqs plot_nonint.py nonint_1orb.h5 nonint_2orb.h5
