import json, glob
from pathlib import Path
import torch
from analysis.circuits.gradient_size_sweep_runner import _apply_sweep_config, _build_mode_method
from circuit.probe_dataset import ProbeDatasetBuilder
from config import config
from data.loader import DataLoader
from eval.ablation_faithfulness import circuit_only_activation, measure_seed_activation, upstream_sites
from eval.floors import collect_site_anchors
from hardware import detect_devices, is_fast_memory, should_compile
from model.inference import Inference
from pipeline.component_index import split_component_idx
from pipeline.discovery_artifacts import load_discovery_artifacts
from sae.bank import SAEBank
RUN_ROOT = Path("/mnt/x/Projects/AIs/Turing/Publication/3 Implementation/Runs/20260531-152059-37117a33/20260531-152059-37117a33")
D1 = Path("experiments/012-driver-bakeoff")
N_SEQ, N_TR, EVAL_BS = 64, 48, 16
load_discovery_artifacts(RUN_ROOT, candidates_path=RUN_ROOT / "candidates.pt")
devices = detect_devices(); device = devices[0]
loader = DataLoader(device=device, pin_memory=is_fast_memory())
inference = Inference(device=device, compile=should_compile())
bank = SAEBank(devices=devices, load_decoders=is_fast_memory(), compile=should_compile())
pb = ProbeDatasetBuilder(inference, bank, loader)
nk = len(bank.kinds)
avg = torch.zeros((bank.n_layer*nk, bank.d_sae), device=bank.device)
_apply_sweep_config(max_per_site=24); config.discovery.eval_batch_size = EVAL_BS
print("| seed | K | free0 (members at natural) | free0 random |")
print("|---|---:|---:|---:|")
import random
for sc, sl in [(13,30053),(25,10628),(26,17432),(35,6599)]:
    layer, ki = split_component_idx(sc, nk); kind = bank.kinds[ki]
    up = upstream_sites(bank, layer, kind); ups = sorted(up)
    m0 = _build_mode_method("counterfactual_gradient","local",inference,bank,avg,pb)
    pd_ = m0.build_probe_dataset(sc, sl); del m0
    pt, pa = pd_.pos_tokens[:N_SEQ], pd_.pos_argmax[:N_SEQ]
    pt_ev, pa_ev = pt[N_TR:], pa[N_TR:]
    a_pos = float(measure_seed_activation(inference,bank,pt_ev,layer,kind,sl,pa_ev,batch_size=EVAL_BS))
    a_e0 = float(circuit_only_activation(inference,bank,{},up,pt_ev,layer,kind,sl,pos_argmax=pa_ev,batch_size=EVAL_BS))
    den = a_pos - a_e0
    _, pins = collect_site_anchors(inference,bank,pt_ev,up,pa_ev,pin_position_specific=False)
    dw = torch.load(D1/("direct_full_%d_%d.pt"%(sc,sl)),map_location="cpu",weights_only=False)["direct"]
    tri=[]
    for s,w in dw.items():
        v,ix = torch.topk(w,k=min(512,w.numel())); tri += [(float(a),s,int(i)) for a,i in zip(v,ix)]
    tri.sort(key=lambda x:-x[0])
    mem_all=[(s,i) for _,s,i in tri if float(pins[s][i])>0 if s in pins]
    rng = random.Random(5)
    for K in (8,16,64):
        mem = mem_all[:K]
        keep={}
        for s,i in mem: keep.setdefault(s,set()).add(i)
        a=float(circuit_only_activation(inference,bank,keep,up,pt_ev,layer,kind,sl,pos_argmax=pa_ev,batch_size=EVAL_BS))
        rnd=[(ups[rng.randrange(len(ups))],rng.randrange(40960)) for _ in range(K)]
        kr={}
        for s,i in rnd: kr.setdefault(s,set()).add(i)
        ar=float(circuit_only_activation(inference,bank,kr,up,pt_ev,layer,kind,sl,pos_argmax=pa_ev,batch_size=EVAL_BS))
        print("| %s L%d %s | %d | %.4f | %.4f |"%("%d/%d"%(sc,sl),layer,kind,K,(a-a_e0)/den,(ar-a_e0)/den))
