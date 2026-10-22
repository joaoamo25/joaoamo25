# Nota de cada configuração da audição, contra a voz REAL da pessoa.
# uso: python3 avalia.py audicao.json "pasta/memos/*.m4a" [--fora id1,id2]   (--fora = memos usados como referência)
# Régua (todas medidas, nada de ouvido):
#   sim     semelhança de locutor (wespeaker) com o centróide dos memos que NÃO viraram referência.
#           Teto prático = um memo real contra os outros; piso = outra pessoa (~0,1).
#   wer     erro de texto pelo ASR (números normalizados por extenso). Errou palavra, take ruim.
#   enton   desvio da entonação em semitons. Abaixo do real = robótico; muito acima = teatral.
#   pal_s   palavras por segundo. Compare com o ritmo real dos memos.
import sys, glob, json, numpy as np, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from voz_lib import *
P = json.load(open(sys.argv[1])); OUT = P.get("saida", "aud"); L = P["linhas"]
fora = sys.argv[sys.argv.index("--fora") + 1].split(",") if "--fora" in sys.argv else []
memos = [f for f in sorted(glob.glob(sys.argv[2])) if not any(u in f for u in fora)]
if len(memos) < 3: sys.exit("preciso de pelo menos 3 memos fora da referência")
E = np.array([emb(le16(f)) for f in memos]); cen = E.mean(0); cen /= np.linalg.norm(cen)
teto = np.mean([float(E[i] @ (lambda c: c / np.linalg.norm(c))(np.delete(E, i, 0).mean(0))) for i in range(len(E))])
def ritmo(x):
    v = np.where(np.abs(x) > 0.02)[0]; return len(norm(asr(x))) / max(0.5, (v[-1] - v[0]) / 16000) if len(v) else 0
real = {"enton": np.mean([prosodia(le16(f))["f0_var_st"] for f in memos]), "pal_s": np.median([ritmo(le16(f)) for f in memos])}
print(f"REAL: teto sim {teto:.3f} · entonação {real['enton']:.2f} st · ritmo {real['pal_s']:.2f} pal/s ({len(memos)} memos)")
res = {}
for nome in P["cfgs"]:
    s, w, pv, ps, erros = [], [], [], [], []
    for i, t in enumerate(L):
        f = f"{OUT}/{nome}_a{i}.wav"
        if not os.path.exists(f) or os.path.getsize(f) < 10000: continue
        x = le16(f); s.append(float(emb(x) @ cen)); h = asr(x); e = wer(t, h); w.append(e)
        pv.append(prosodia(x)["f0_var_st"]); ps.append(ritmo(x))
        if e > 0.05: erros.append(f"   a{i} wer {e:.2f}: {h}")
    if not s: continue
    r = {"sim": round(np.mean(s), 3), "wer": round(np.mean(w), 3), "enton": round(np.mean(pv), 2), "pal_s": round(np.mean(ps), 2), "n": len(s)}
    # nota única: identidade manda; erro de texto e prosodia fora do real descontam
    r["nota"] = round(r["sim"] - 0.5 * r["wer"] - 0.02 * abs(r["enton"] - real["enton"]) - 0.03 * abs(r["pal_s"] - real["pal_s"]), 3)
    res[nome] = r
for nome, r in sorted(res.items(), key=lambda kv: -kv[1]["nota"]):
    print(f"{nome:22} nota {r['nota']:.3f} | sim {r['sim']:.3f} wer {r['wer']:.2f} enton {r['enton']:.2f} pal/s {r['pal_s']:.2f} (n={r['n']})")
json.dump({"real": {"teto_sim": round(teto, 3), **{k: round(float(v), 2) for k, v in real.items()}}, "cfgs": res}, open(f"{OUT}/notas.json", "w"), indent=1)
