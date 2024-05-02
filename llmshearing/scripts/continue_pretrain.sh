# pruning moss2 7b -> 3b or 1.3b

PROJ_DIR="/remote-home/xjzhao/LLM_Shearing/llmshearing"
DATA_DIR=/remote-home/share/personal/zyzeng/data/moss2.5b_sampled/for_prune
OUTPUT_DIR=${PROJ_DIR}/ckpts/
LAUNCH_SCRIPT=${PROJ_DIR}/llmshearing/scripts/launch.sh
TRAIN_SCRIPT=${PROJ_DIR}/llmshearing/train.py

test=True

model=100m # target model size
config_file=${PROJ_DIR}/llmshearing/configs/internlm/${model}.yaml
prune_run_name=continue_pretrain_${model}_sl4096
# path=${OUTPUT_DIR}/${prune_run_name}/pruned-latest-rank0.pt # path to the 
path=${OUTPUT_DIR}/moss_2.5b_pruning_scaling_constant_to100m_sl4096_run0/latest-rank0.pt

# data setup
data_local=${DATA_DIR}

# basic setup
max_seq_len=4096
device_train_microbatch_size=8
global_train_batch_size=384
device_eval_batch_size=8

# learning setup
lr=1e-4 # learning rate for the main parameters
max_duration=48000ba # 50B tokens
save_interval=3200ba # save every 3200ba
t_warmup=1440ba # 3% learning rate warmup 

# dynamic loading setup
dynamic=True
set_names=[arxiv,cn_baike,cn_book,cn_weixin,cn_zhihu,code_starcoder,en_book,en_stackexchange,wanjuan,wiki] # domain names
proportion=[0.0038885832078429925,0.013171063389765087,6.913036813943098e-06,0.11492975550956504,0.06456880079775063,0.16964341743831857,0.0005333269641220821,0.07731814128971792,0.4785982974469239,0.07734170091917984] # initial proportion of RP, make sure that the sum(proportion) = 1
# doremi: update weights with exponential descent
# constant: keep the weights constant
update_type=constant
target_loss=[1.2465803623199463,3.0522592067718506,2.5978455543518066,2.5219357013702393,3.025589942932129,1.0684151649475098,2.2368016242980957,1.7007801532745361,2.288639783859253,1.634076714515686] # 410m predicted loss from scaling law

eval_split_name=eval_merge # eval on all domains
eval_interval=400ba # eval every 50 batches and update the loading proportion


# save directroy
run_name=${prune_run_name}_ft${max_duration}_v3
save_dir=${OUTPUT_DIR}/finetune/${run_name}


# Run with slurm
sbatch -p moss \
    --job-name ${run_name} \
    --nodes=6 \
    --gpus-per-node=8 \
    --mem=512gb \
    --cpus-per-task=8 \
    --output=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/finetune/finetune_%A_%a.out \
    --error=/remote-home/zyzeng/LLM-Shearing/LLM-Shearing/logs/finetune/finetune_%A_%a.err \
    $LAUNCH_SCRIPT \
    $config_file \
    run_name=${run_name} \
    data_local=${data_local} \
    eval_loader.dataset.split=${eval_split_name} \
    eval_loader.dataset.num_canonical_nodes=${global_train_batch_size} \
    train_loader.dataset.num_canonical_nodes=${global_train_batch_size} \
    global_train_batch_size=${global_train_batch_size} \
    device_train_microbatch_size=${device_train_microbatch_size} \
    device_eval_batch_size=${device_eval_batch_size} \
    max_seq_len=${max_seq_len} \
    max_duration=${max_duration} \
    eval_first=true \
    scheduler.t_warmup=${t_warmup} \
    save_folder=${save_dir} \
    save_overwrite=False \
    eval_interval=${eval_interval} \
    save_interval=${save_interval} \
    optimizer.lr=${lr} \
    model.l0_module=null \
    model.path=${path} \
    callbacks.data_loading.dynamic=${dynamic} \
    callbacks.data_loading.set_names=${set_names} \
    callbacks.data_loading.proportion=${proportion} \
    callbacks.data_loading.update_type=${update_type} \
    callbacks.data_loading.target_loss=${target_loss} \
    train_loader.num_workers=0 \
    train_loader.prefetch_factor=null \
    train_loader.persistent_workers=false \
    autoresume=true

# checking eval_first