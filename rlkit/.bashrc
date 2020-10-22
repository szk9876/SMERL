export PYTHONPATH=$PYTHONPATH:$(pwd)
export PYTHONPATH=$PYTHONPATH:/iris/u/szk/SMERL_paper_NeurIPS/SMERL_codebase/multiworld
export PYTHONPATH=$PYTHONPATH:/iris/u/szk/SMERL_paper_NeurIPS/SMERL_codebase/viskit
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/sailhome/szk/.mujoco/mujoco200/bin
# export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/afs/cs.stanford.edu/u/szk/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/nvidia-384/libGL.so
# export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libGLEW.so
source /iris/u/szk/anaconda3/bin/activate
source activate rlkitfinalbaby
