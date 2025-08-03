# Innovation and exnovation dynamics on trees and trusses
Authors: [Edward D. Lee](https://eddielee.co) and Ernesto Ortega-Díaz

For code corresponding to "Idea engines: Unifying innovation and obsolescence from markets and genetic evolution to science" see release [v0.1.0](https://github.com/eltrompetero/innovation/releases/tag/pnas). The featured branch is now dedicated to a new project.

This repository is for code to accompany "Innovation-exnovation dynamics on trees and trusses" published in PRR.

Lee, E. D. & Ortega-Díaz, E. Innovation-exnovation dynamics on trees and trusses. Phys. Rev. Research 7, 033102 (2025). [https://journals.aps.org/prresearch/abstract/10.1103/ynwt-7g91](https://journals.aps.org/prresearch/abstract/10.1103/ynwt-7g91).

## Structure
'pipeline_notebook.ipynb' contains figures that go into the paper and contains references to the code that
must be run to render the figures as well as cached files that are necessary. These are generated in
'innov/pipeline.py'.

'scripts' directory contains loops for computing the replica survival probabilities over the space of
dynamical and structural parameters.

## Dependencies
Code works with Python 3.12. See `spec-file.txt` for necessary libraries. Required hardware: Nvidia GPUs.
