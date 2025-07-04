# Recommended Execution:
## First generate your MuJoCo simulation data:

```
python finetune_rl.py --output dataset/ --model mujoco/cdpr.xml
```

## Prepare your prompt dataset (JSON file with natural language commands)

## Run training with your chosen RL algorithm:

```
python finetune_rl.py --rl_algorithm PPO --mujoco_model_path mujoco/cdpr.xml
```