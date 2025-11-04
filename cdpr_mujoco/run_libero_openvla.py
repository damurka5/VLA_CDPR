from cdpr_libero_adapter import CDPRLiberoEnv
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
import numpy as np
from PIL import Image

def make_openvla(model_name="openvla/openvla-7b", device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
    ).to(device)
    return processor, vla, device

def vla_action(proc, vla, device, rgb, instruction="go near the red square"):
    prompt = f"In: What action should the robot take to {instruction}?\nOut:"
    img = Image.fromarray(rgb)
    inputs = proc(prompt, img).to(device, dtype=torch.bfloat16)
    act = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
    if isinstance(act, torch.Tensor):
        act = act[:3].detach().float().cpu().numpy()
    elif isinstance(act, (list, tuple, np.ndarray)):
        act = np.asarray(act)[:3].astype(np.float32)
    else:
        act = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    return act

def main():
    env = CDPRLiberoEnv(xml_path="VLA_CDPR/mujoco/cdpr.xml", egl=True)
    obs = env.reset(bddl_problem_path="/root/repo/LIBERO/libero/libero/bddl_files/cdpr_libero/cdpr_problem.bddl")

    proc, vla, device = make_openvla()
    instr = "go near the red square"

    for t in range(800):
        rgb = obs["rgb"]
        action = vla_action(proc, vla, device, rgb, instr)
        obs, rew, done, info = env.step(action)
        if (t % 50) == 0:
            print(f"t={t} reward={rew:.3f} dist={info['dist_to_goal']:.3f}")
        if done:
            print(f"✓ Success at t={t}")
            break

    env.close()

if __name__ == "__main__":
    main()
