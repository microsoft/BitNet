### for training fp16 baseline

lrs=("5e-6")
model="Qwen3-1.7B"

for idx in "${!lrs[@]}"; do
  lr="${lrs[$idx]}"
  gpu="$((idx))"

  echo "Launching LR=$lr on GPU $gpu..."

  sudo docker run --privileged --net=host --ipc=host --rm \
    -v $(pwd):$(pwd) -w $(pwd) \
    rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.5.1 \
    bash -c "
      bash mi300_setup.sh && bash run_qwen3_sft_distill.sh $lr $model $gpu
    " &

done

wait
echo "All jobs finished."


### for training bitdistill by using fp16 baseline as teacher

lrs=("5e-6")
model="Qwen3-1.7B"

for idx in "${!lrs[@]}"; do
  lr="${lrs[$idx]}"
  gpu="$((idx))"

  echo "Launching LR=$lr on GPU $gpu..."

  sudo docker run --privileged --net=host --ipc=host --rm \
    -v $(pwd):$(pwd) -w $(pwd) \
    rocm/pytorch:rocm6.4.1_ubuntu24.04_py3.12_pytorch_release_2.5.1 \
    bash -c "
      bash mi300_setup.sh && bash run_qwen3_sft_distill.sh $lr $model $gpu <PUT-TEACHER_MODEL-PATH-HERE> 1.0 1e5 24
    " &

done

wait
echo "All jobs finished."
