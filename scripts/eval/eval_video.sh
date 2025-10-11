#!/bin/bash
MODEL_PATH=${1}
BENCHMARKS=${2:-"videomme,egoschema,nextqa,egoperceptionmcq,egoperceptionmcq_depth,adlx_mcq,adlx_descriptions"}

ARG_WORLD_SIZE=${3:-1}
ARG_NPROC_PER_NODE=${4:-8}

if [[ -v MASTER_ADDR_PASSED ]]; then
    ARG_MASTER_ADDR=$MASTER_ADDR_PASSED # passed via slurm submission script
else
    ARG_MASTER_ADDR=127.0.0.1 # for dev environments
fi
ARG_MASTER_PORT=12355
ARG_RANK=$SLURM_NODEID

if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"
echo "MODEL_PATH: $MODEL_PATH"
echo "BENCHMARKS: $BENCHMARKS"


SAVE_DIR=local_evaluations/$(basename $MODEL_PATH)
DATA_ROOT=/path/to/vlm_eval_bench
declare -A DATA_ROOTS

# save command used to run this script
mkdir -p "${SAVE_DIR}"
CMD="$(ps -o args= -p $$)"
echo "$CMD" > "$SAVE_DIR/run_command"

# mcqa
DATA_ROOTS["videomme"]="$DATA_ROOT/videomme"
DATA_ROOTS["egoschema"]="$DATA_ROOT/egoschema"
DATA_ROOTS["nextqa"]="$DATA_ROOT/nextqa"
DATA_ROOTS["egoperceptionmcq"]="$DATA_ROOT/egoperceptionmcq"
DATA_ROOTS["egoperceptionmcq_depth"]="$DATA_ROOT/egoperceptionmcq"
DATA_ROOTS["adlx_mcq"]="$DATA_ROOT/adlx"
DATA_ROOTS["adlx_descriptions"]="$DATA_ROOT/adlx"

IFS=',' read -ra BENCHMARK_LIST <<< "$BENCHMARKS"
for BENCHMARK in "${BENCHMARK_LIST[@]}"; do
    DATA_ROOT=${DATA_ROOTS[$BENCHMARK]}
    if [ -z "$DATA_ROOT" ]; then
        echo "Error: Data root for benchmark '$BENCHMARK' not defined."
        continue
    fi

    # If you cant install ollama locally (e.g. on a cluster), you can run an ollama server to perform the evaluations for benchmarks that require it
    if [[ "$BENCHMARK" =~ ^(egoperceptionmcq|egoperceptionmcq_depth|adlx_mcq|adlx_descriptions)$ ]] && ! pgrep -f "ollama serve" > /dev/null; then
        nohup /path/to/ollama_server/bin/ollama serve > /dev/null 2>&1 &
        sleep 2
    fi
    
    torchrun --nnodes $WORLD_SIZE \
        --nproc_per_node $NPROC_PER_NODE \
        --master_addr=$MASTER_ADDR \
        --master_port=$MASTER_PORT \
        --node_rank $RANK \
        evaluation/evaluate.py \
        --model_path ${MODEL_PATH} \
        --benchmark ${BENCHMARK} \
        --data_root ${DATA_ROOT} \
        --save_path "${SAVE_DIR}/${BENCHMARK}.json" \
        --fps 1 \
        --max_frames 180 \
        --max_visual_tokens 16384 \
        --num_workers 4
done