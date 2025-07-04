# Recommended Execution:
## First generate your MuJoCo simulation data:

```
python generate_mujoco_data.py --output dataset/ --model cdpr_model.xml
```

## Prepare your prompt dataset (JSON file with natural language commands)

## Run training with your chosen RL algorithm:

```
python train_rl_openvla.py --rl_algorithm PPO --mujoco_model_path cdpr_model.xml
```