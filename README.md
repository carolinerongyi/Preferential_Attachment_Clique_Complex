## Introduction

This repo provides codes to investigate the Betti numbers of preferential attachment clique complexes. In our paper (see [below](#how-to-cite?)), we determined the asymptotics of the expected Betti numbers. In this package, we provide the codes to generate the graphs, compute their Betti numbers, and compute their estimates in our paper. We also provide a jupyter notebook to illustrate our codes.

## How to cite?

C. Siu, G. Samorodnitsky, C. Yu, and R. He, The Asymptotics of the Expected Betti Numbers of Preferential Attachment Clique Complexes, 2023.

The arxiv link is [https://arxiv.org/abs/2305.11259](https://arxiv.org/abs/2305.11259).

The bibtex entry is as follows.

`@misc{siu2023asymptotics,
      title={The Asymptotics of the Expected Betti Numbers of Preferential Attachment Clique Complexes}, 
      author={Chunyin Siu and Gennady Samorodnitsky and Christina Lee Yu and Rongyi He},
      year={2023},
      eprint={2305.11259},
      archivePrefix={arXiv},
      primaryClass={math.PR}
}`

## Codes
1. 'simulator_pa.py' contains functions to generate a peferential attachment model using iGraph
2. 'betti.py' contains functions to measure the topological features of a complex
3. 'tutorial.ipynb' is a Jupyter Notebook that demonstrates the simulations used in the paper.

## Requirement
The project depends on the following packages in Python:
- Ripser
- NumPy
- iGraph
- Numba
- Matplotlib
- Itertools

## Authors and Contact

* [corresponding author] Chunyin Siu (Alex), Center for Applied Mathematics, Cornell University, NY, USA (cs2323 [at] cornell.edu)

* Gennady Samorodnitsky, School of Operations Research and Information Engineering, Cornell University, NY, USA (gs18 [at] cornell.edu)

* Christina Lee Yu, School of Operations Research and Information Engineering, Cornell University, NY, USA (cleeyu [at] cornell.edu)

* Rongyi He, School of Operations Research and Information Engineering, Cornell University, NY, USA, NY, USA (rh4643 [at] cornell.edu)



