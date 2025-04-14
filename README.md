# Online PbRL

This implementation is based on the official codebase of [B-Pref](https://github.com/rll-research/BPref). It features state-of-the-art online PbRL algorithms, including PrefPPO, PEBBLE, RUNE, SURF, MRN, and QPA, along with SAC, which serves as the oracle.



## How to install

```bash
conda env create -f environment.yml
cd custom_dmc2gym/
pip install -e .
```


## How to run

To train a model, simply run the scripts in `./scripts/[task]/run_[algo].sh` after activating the environment. For example, train a policy using PEBBLE algorithm for DeepMind control suite task Walker_Walk:

```bash
./scripts/walker_walk/run_pebble.sh
```

Train a policy using QPA algorithm for Metaworld task Door Open:
```bash
./scripts/metaworld_door-open-v2/run_qpa.sh
```

The hyperparameters for different algorithms are referenced from the corresponding papers or official implementations. The running device can be changed by modifying the `CUDA_VISIBLE_DEVICES` variable in each bash script.
