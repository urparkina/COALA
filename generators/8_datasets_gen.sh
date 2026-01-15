datasets=('BoolQ' 'PIQA' 'SIQA' 'hellaswag' 'winogrande' 'ARC-E' 'ARC-C' 'OBQA')

for dataset in "${datasets[@]}"; do
    python3 -u ./generators/generate_common_reasoning.py $1 ./datasets/"$dataset" -s "$dataset"
done