# CPSC 554X - Group Project

## Team Members
- [Gengran Li](https://github.com/CcccLllll)
- [Rongshuai Wu](https://github.com/MrRhine98)
- [Yisheng Liu](https://github.com/liuyishengalan)

## Project Description
We proposed to feed a neural network time-domain data generated from the FDTD model along with its ground-truth area function. Hopefully, the neural network will be able to learn the mapping scheme and generate the corresponding geometry from unknown time-domain data

## Project Structure
```
.
CPSC554X_project/
├── 1segtube
│   ├── dataset
│   │   ├── acoustic_data.txt
│   │   ├── audio_1.mat
│   │   └── geometry_data.txt
│   ├── generate_dataset.py
│   └── mlp.py
├── 2segtube
│   ├── dataset
│   │   ├── acoustic_data.txt
│   │   └── geometry_data.txt
│   ├── generate_dataset.py
│   └── mlp.py
└── README.md
```
## Milestone
- [x] 1. Generate dataset
- [x] 2. Train a neural network
- [x] 3. Test the neural network
- [x] 4. Generate a 2D geometry from unknown time-domain data
- [x] 5. Write a report