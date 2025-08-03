# Innovation and Exnovation Dynamics on Trees and Trusses

**Authors:** [Edward D. Lee](https://eddielee.co) and Ernesto Ortega-Díaz

This repository accompanies the article:

Lee, E. D. & Ortega-Díaz, E. Innovation-exnovation dynamics on trees and trusses. *Phys. Rev. Research* **7**, 033102 (2025). [Journal link](https://journals.aps.org/prresearch/abstract/10.1103/ynwt-7g91)

## Overview
This repository provides code and data for reproducing the results and figures in the above publication. For code related to our previous work, see release [v0.1.0](https://github.com/eltrompetero/innovation/releases/tag/pnas).

## Directory Structure
- `innov/pipeline.py`: Main code for generating figures and results.
- `pipeline_notebook.ipynb`: Jupyter notebook for rendering figures and referencing code and cached files.
- `scripts/`: Contains scripts for computing replica survival probabilities across parameter spaces.
- `spec-file.txt`: Lists required Python packages.

## Getting Started
1. **Clone the repository:**
   ```sh
   git clone https://github.com/eltrompetero/innovation.git
   cd innovation
   ```
2. **Install dependencies:**
   - Python 3.12 required.
   - Install packages from `spec-file.txt`:
     ```sh
     pip install -r spec-file.txt
     ```
   - Required hardware: Nvidia GPU.
3. **Run scripts:**
   - See `scripts/` for example usage and parameter sweeps.
   - Use `pipeline_notebook.ipynb` to reproduce figures from the paper.

## Citation
If you use this code, please cite our article:
```
@article{lee2025innovation,
  title={Innovation-exnovation dynamics on trees and trusses},
  author={Lee, Edward D. and Ortega-Díaz, Ernesto},
  journal={Phys. Rev. Research},
  volume={7},
  pages={033102},
  year={2025}
}
```

## Contact
For questions or issues, please open an issue on GitHub or contact the authors via their websites.

## License
See `LICENSE` for terms of use.
