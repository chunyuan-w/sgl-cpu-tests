cd /home/chunyuan/heteroflow/sgl-cpu-tests
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so

numactl --physcpubind=0-27 --membind=0 python test_moe_int8.py