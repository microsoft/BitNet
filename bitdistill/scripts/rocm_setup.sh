
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/rocm6.2.4

pip install -e ".[torch,metrics]"

export NCCL_TIMEOUT=18000
pip install liger-kernel

cd tools/transformers
pip install -e .
cd ..
cd bitnet
pip install -e .
cd ..
pip install --upgrade accelerate

echo "BEGIN INSTALL PYTHON"
apt-get update
apt-get install python3.10-dev -y
echo "PYTHON END"

pip install --upgrade boto3 botocore
pip install datasets
pip install wandb
pip3 install deepspeed==0.16.0
pip install accelerate==1.7.0
python3 -m pip install --upgrade 'optree>=0.13.0'