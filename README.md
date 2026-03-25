# Cross-chain Bridge Sigmetrics Artifact

Data and analysis code for the paper "The Price of Interoperability: Exploring Cross-Chain Bridges
           and Their Economic Consequences".

## Structure

```text
cross-chain-bridge-sigmetrics-artifact/
  code/
    experiments.py
    net_flow.py
    plot_bridge_share_and_value.py
    chain_share_figGen.py
  data/
    bridge_data_final.csv
    chain_attributes_final.csv
```

## Run scripts

1. Full pipeline: `code/experiments.py`

```bash
python code/experiments.py \
  --input data/bridge_data_final.csv \
  --outdir out_experiments \
  --start_date 2022-01-01 \
  --cutoff 2025-10-31 \
  --topk_bridge 10 \
  --topk_chain 10 \
  --steps all
```

2. Net-flow figures: `code/net_flow.py`

```bash
python code/net_flow.py \
  --input data/bridge_data_final.csv \
  --outdir out_netflow \
  --cutoff 2025-10-31 \
  --top_corridors 12
```

3. Bridge share & value: `code/plot_bridge_share_and_value.py`

```bash
python code/plot_bridge_share_and_value.py \
  --input data/bridge_data_final.csv \
  --outdir out_bridge_share
```

4. Chain endpoint share: `code/chain_share_figGen.py`

This script has no CLI args and expects `bridge_data_final.csv` to be reachable from the current working directory:

```bash
cd cross-chain-bridge-sigmetrics-artifact
ln -sf data/bridge_data_final.csv ./bridge_data_final.csv
cd code
python chain_share_figGen.py
```


