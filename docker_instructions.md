docker build -t bitnet-inference-api . --progress plain

docker run --rm -p 11435:11435 bitnet-inference-api
