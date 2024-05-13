# Projects on innovation and obsolescence.

For code corresponding to "Idea engines: Unifying innovation and obsolescence from markets and genetic evolution to science" see release [v0.1.0](https://github.com/eltrompetero/innovation/releases/tag/pnas). The main branch is now dedicated to a new project.

## Innovation and exnovation dynamics on trees and trusses
Code to accompany simple innovation/obsolescence model by Ernesto Ortega, Niraj Kushwaha, and Edward D. Lee.

automaton_exnovation.py: automaton
network_model_1_SDE_jax_obs_front.py: ODE and SDE numerical integration
simple_calculations.py: analytics and numerics of compartamental pseudogap approximations
tree_gen.py: Graph generator


## Dependencies
Code works with Python 3.12.

```bash
conda install jaxlib=*=*cuda* jax cuda-nvcc numpy scipy jupyter ipython matplotlib networkx hickle dill statsmodels pygraphviz tensorflow tensorboard-plugin-profile -c conda-forge -c nvidia
```