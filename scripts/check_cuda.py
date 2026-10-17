"""Check CPU and CUDA tensors and the official full actor-critic; no game training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from reproduction.runtime import ROOT, load_config, static_spaces, seed_all, metadata, write_json
from neroRL.nn.actor_critic import create_actor_critic_model


def main():
    seed_all(0)
    report = metadata()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GPU validation has NOT passed")
    if "4060" not in torch.cuda.get_device_name(0):
        raise AssertionError("Expected RTX 4060")
    report["devices"] = {}
    spaces, actions = static_spaces()
    config = load_config(ROOT / "configs/smoke.yaml")
    for device in ("cpu", "cuda"):
        x = torch.randn(32,32,device=device,requires_grad=True)
        (x @ x.T).square().mean().backward()
        assert torch.isfinite(x.grad).all()
        model = create_actor_critic_model(config["model"],spaces,None,actions,torch.device(device))
        obs = {k:torch.rand((2,*space.shape),device=device) for k,space in spaces.spaces.items()}
        policy,value,_ = model(obs)
        selected = policy[0].sample()
        loss = -policy[0].log_prob(selected).mean() + value.square().mean()
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().max().item()>0 for g in gradients)
        assert policy[0].probs.shape==(2,7) and value.shape==(2,)
        report["devices"][device] = {"tensor_backward":"passed","official_network_forward_backward":"passed",
            "parameters":sum(p.numel() for p in model.parameters()),"loss":float(loss.detach().cpu()),"synthetic_inputs":True}
    torch.cuda.synchronize()
    report["real_game_training"] = False
    write_json(ROOT / "artifacts/cuda-validation.json",report)
    print(report)


if __name__ == "__main__":
    main()
