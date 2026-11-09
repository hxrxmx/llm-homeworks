export HF_HOME="${HF_HOME:-$PWD/cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$PWD/cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$PWD/cache/triton}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
export MAX_JOBS=4

cuda_dir=$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"] + "/nvidia/cu13")')
if [ -x "$cuda_dir/bin/nvcc" ]; then
    export CUDA_HOME="$cuda_dir"
    export PATH="$CUDA_HOME/bin:$PATH"
    export LD_LIBRARY_PATH="$CUDA_HOME/lib:${LD_LIBRARY_PATH:-}"
fi
