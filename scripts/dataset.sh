CONFIG=$1

./generators/common_reasoning_gen.sh ./scripts/configs/$CONFIG.yaml
./generators/8_datasets_gen.sh ./scripts/configs/$CONFIG.yaml
./generators/wiki_gen.sh ./scripts/configs/$CONFIG.yaml
