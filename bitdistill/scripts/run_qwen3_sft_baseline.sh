LR=$1
MODEL=$2
GPU=$3
YAML="$(pwd)/yamls/training_args/sft_nli.yaml"
RUN_NAME="qnli-${MODEL}-bitnet-allnorm-warm0.05-epoch2-lr${LR}-bs32xa1xg1-wd0.1-mi300"
OUTPUT_ROOT_DIR=<PUT_OUTPUT_ROOT_PATH_HERE>
OUTPUT_DIR="${OUTPUT_ROOT_DIR}/${RUN_NAME}"
mkdir -p ${OUTPUT_DIR}

HIP_VISIBLE_DEVICES=${GPU} llamafactory-cli train ${YAML} per_device_train_batch_size=32 \
      run_name=${RUN_NAME} \
      output_dir=${OUTPUT_DIR} \
      model_name_or_path=Qwen/${MODEL} \
      gradient_accumulation_steps=1 \
      learning_rate=${LR} \
      num_train_epochs=2.0 \
      save_steps=2000 eval_steps=200 \
      per_device_eval_batch_size=128 \
      template=qwen3 \
      weight_decay=0.1 \
      lr_scheduler_type=linear \
      dataset_dir=/mnt/msranlp/xun/BieNet_finetuning/datasets/qnli \
      dataset=qnli_train eval_dataset=qnli_valid \
      2>&1 | tee $OUTPUT_DIR/train.log
