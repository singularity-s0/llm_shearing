MODEL_DIR=/remote-home/xjzhao/LLM_Shearing/llmshearing/ckpts/moss_2.5b_pruning_scaling_constant_to100m_sl4096_no_freeze_emb
MODEL_PATH=$MODEL_DIR/pruned-latest-rank0.pt
OUTPUT_PATH=$MODEL_DIR/hf-pruned-latest_rank0
MODEL_CLASS=Moss2ForCausalLM
HIDDEN_SIZE=1024
NUM_ATTENTION_HEADS=8
NUM_HIDDEN_LAYERS=5
INTERMEDIATE_SIZE=4096
MODEL_NAME=/remote-home/share/models/moss2-2_5b-hf/

python3 -m llmshearing.utils.composer_to_hf $MODEL_PATH $OUTPUT_PATH \
        model_class=${MODEL_CLASS} \
        hidden_size=${HIDDEN_SIZE} \
        num_attention_heads=${NUM_ATTENTION_HEADS} \
        num_hidden_layers=${NUM_HIDDEN_LAYERS} \
        intermediate_size=${INTERMEDIATE_SIZE} \
        num_key_value_heads=${NUM_ATTENTION_HEADS} \
        _name_or_path=${MODEL_NAME}
