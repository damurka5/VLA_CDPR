# VLA_CDPR
VLA models application in CDPR

## LLM Perspective

- **Model Selection**  
  Tested different Vision-Language-Action (VLA) models and selected OpenVLA as the base architecture

- **Dataset Preparation**  
  Created text dataset (prompts to language model) specifically for Cable-Driven Parallel Robot (CDPR) control  
  Location: `dataset/`

- **Simulation Environment**  
  Developed MuJoCo model of CDPR with end-effector camera  
  Location: `mujoco/cdpr.xml`

- **Training Infrastructure**  
  Implemented code to train VLA directly from MuJoCo simulation data  
  Location: `mujoco/`

- **Fine-Tuning Implementation**  
  Prepared specialized fine-tuning code for OpenVLA's language model (Llama2-based)  
  Location: `tuning/Llama2_tuning.py`

## AgenticAI Perspective

- **RL Fine-Tuning**  
  Developed reinforcement learning pipeline for optimizing OpenVLA's CDPR control  
  Location: `tuning/finetune_rl.py`

- **Simulation Integration**  
  Enhanced MuJoCo CDPR model with real-time simulation capabilities  
  Location: `mujoco/run_mujoco.py`

- **Safety System**  
  Implemented safety agent that evaluates motion requests for potential hazards  
  Location: `safety/agent.py`

- **Safety Training**  
  Created training script for safety agent's classification capabilities  
  Integrated within: `safety/agent.py`

## **Problem**

- Standard VLA (Video-Language-Action) models lack physical understanding of concepts like mass, gravity, and inertia.
- This limits generalization to new object interactions or environments in real-world robotic applications.

## **Our Expected Contribution**

- Develop a modular VLA model that integrates differentiable physics to simulate and predict real-world object dynamics.
- Learn physical parameters (e.g., mass, velocity, gravity effects) from raw video and language inputs via an interpretable latent space.
- Enable the robot to reason about physical laws and plan robust manipulation strategies in changing conditions.
- Ground visual and language concepts (e.g., "heavy", "fall", "support") to latent physical parameters for enhanced action selection.

## Problem Statement

The project aims to deploy a cable-driven parallel robot (CDPR) to maintain the equilibrium of various objects on a moving platform. The challenge is to control the CDPR effectively to stabilize these objects despite platform motion, comparing the efficiency of different control and modeling approaches including VLA models, other machine learning models, and nonlinear optimization techniques for actuator control.

## Keywords

- Cable-Driven Parallel Robot (CDPR)
- Variable Latent Attention (VLA) model
- Robot equilibrium control
- Nonlinear optimization
- Actuator control
- Dynamic modeling
- Real-time control
- Multi-cable tension distribution

## Possible Solution Paths

- Application of VLA models to learn and predict control strategies for maintaining object equilibrium on the moving platform.
- Use of alternative machine learning models or classical control algorithms for comparison.
- Formulation and solution of nonlinear optimization problems to compute cable tensions and control actuator inputs ensuring stability and positive cable tensions.
- Simulation and dynamic modeling of the CDPR system to validate control approaches before real-world deployment.

## End Goal

To successfully operate the real CDPR to hold various objects balanced on a moving platform by comparing and evaluating the effectiveness of VLA-based control against other methods, ultimately identifying the most efficient and robust approach for real-time control of cable tensions and platform stability.

## Milestones

- Weeks 1–2: Literature review on CDPR control, VLA models, and nonlinear optimization methods; set up datasets from the real CDPR system.
- Weeks 3–6: Implement VLA model for CDPR control and train on collected datasets.
- Weeks 7–10: Develop and implement baseline control models including nonlinear optimization and classical control.
- Weeks 11–12: Perform comparative analysis of VLA and other models in simulation and on the real robot.
- Week 13: Validate final control approach on the real CDPR with various objects.
- Week 14: Document results and prepare project report.

## Tools & Libraries

- PyTorch + Lightning for implementing and training VLA and other machine learning models.
- OpenVLA framework for applying Variable Latent Attention models.
- OpenAI API or HuggingFace Transformers for additional model support if needed.
- ClearML or Weights & Biases for experiment tracking and training monitoring.

## References

- OpenVLA: https://arxiv.org/abs/2406.09246

- Sammarchi, E. "Dynamic modelling and simulation of a cable-driven parallel robot" (2018) — detailed theoretical background on CDPR kinematics, statics, dynamics, and control including nonlinear optimization methods for cable tension distribution[3](https://amslaurea.unibo.it/id/eprint/17526/1/sammarchi_enrico_tesi.pdf).
- Other literature on CDPR design, control, and applications from sources such as ScienceDirect, MDPI, and academic theses.

