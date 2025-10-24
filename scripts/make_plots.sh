ENTITY="$1"
PROJECT="${2:-cs6886-vgg6-cifar10}"

python MakePlots.py --entity "$ENTITY" --project "$PROJECT" --top_k 50 --outdir plots
