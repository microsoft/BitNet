pip install -e ".[torch,metrics]"

export NCCL_TIMEOUT=18000
pip install liger-kernel

cd tools/transformers
pip install -e .
cd ..
cd bitnet
# cd matmulfreellm2
pip install -e .
cd ..
# pip install --upgrade accelerate
pip install accelerate==1.7.0

echo "BEGIN INSTALL PYTHON"
apt-get update
apt-get install python3.10-dev -y
echo "PYTHON END"
python3 -m pip install --upgrade 'optree>=0.13.0'
