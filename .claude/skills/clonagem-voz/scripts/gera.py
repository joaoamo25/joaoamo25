# Produção: gera N takes por fala com a configuração vencedora da audição e escolhe o melhor por medida.
# uso: python3 gera.py roteiro.json
# roteiro.json: {"pasta": "takes", "memos": "uploads/*.m4a", "fora": ["id_ref"], "takes": 3,
#                "cfg": {"ref": "r.wav", "ref_text": "...", "extra": {"speed": 1.1}},
#                "falas": ["frase 1", "frase 2", ...]}
# Escolha: entre os takes com wer <= 0,06 (1 palavra em ~17, quase sempre grafia do ASR) fica o de maior semelhança; se nenhum passa, o de menor wer
# (e a fala sai marcada "REVISAR" — reescreva a frase em vez de aceitar palavra trocada).
# Saída: <pasta>/escolha.json, já no formato de "segmentos" do monta.py.
import json, sys, os, glob, numpy as np, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from voz_lib import *
R = json.load(open(sys.argv[1])); D = R.get("pasta", "takes"); os.makedirs(D, exist_ok=True)
memos = [f for f in sorted(glob.glob(R["memos"])) if not any(u in f for u in R.get("fora", []))]
cen = np.mean([emb(le16(f)) for f in memos], 0); cen /= np.linalg.norm(cen)
aud = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audicao.py")
cfgs = {f"s{k}": {**R["cfg"], "extra": {**R["cfg"].get("extra", {}), "seed": 7 + 4 * k}} for k in range(R.get("takes", 3))}
json.dump({"saida": D, "linhas": R["falas"], "cfgs": cfgs}, open(f"{D}/_audicao.json", "w"), ensure_ascii=False)
subprocess.run(["python3", aud, f"{D}/_audicao.json"], check=True)
segs = []
for i, t in enumerate(R["falas"]):
    cand = []
    for k in cfgs:
        f = f"{D}/{k}_a{i}.wav"
        if not os.path.exists(f) or os.path.getsize(f) < 10000: continue
        x = le16(f); h = asr(x); cand.append({"arquivo": f, "sim": round(float(emb(x) @ cen), 3), "wer": round(wer(t, h), 3), "ouvido": h})
    if not cand: print(f"fala {i}: nenhum take"); continue
    bons = [c for c in cand if c["wer"] <= 0.06]
    m = max(bons, key=lambda c: c["sim"]) if bons else min(cand, key=lambda c: (c["wer"], -c["sim"]))
    print(f"fala {i}: {os.path.basename(m['arquivo'])} sim {m['sim']} wer {m['wer']}" + ("" if bons else "  REVISAR: " + m["ouvido"]))
    segs.append({"arquivo": m["arquivo"], "texto": t, "voz": "clone", "sim": m["sim"], "wer": m["wer"], **({} if bons else {"revisar": True})})
json.dump({"segmentos": segs}, open(f"{D}/escolha.json", "w"), ensure_ascii=False, indent=1)
