set -e
MODELS_DIR=${1:-"/model/triton_model_repository"}

echo "Starting Triton Inference Server"
echo "Models directory: $MODELS_DIR"

tritonserver \
    --model-repository="$MODELS_DIR" \
    --model-control-mode=explicit \
    --load-model=nvt_user_transform \
    --load-model=nvt_item_transform \
    --load-model=nvt_context_transform \
    --load-model=multimodal_embedding_lookup \
    --load-model=query_tower \
    --load-model=faiss_retrieval \
    --load-model=dlrm_ranking \
    --load-model=item_id_decoder \
    --load-model=feast_user_lookup \
    --load-model=feast_item_lookup \
    --load-model=filter_seen_items \
    --load-model=softmax_sampling \
    --load-model=context_preprocessor \
    --load-model=unroll_features \
    --load-model=ensemble_model