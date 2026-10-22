# Monta a narração: falas clonadas escolhidas (gera.py) + falas reais, com pausas curtas e sem o "estalo" do clone.
# uso: python3 monta.py plano.json
# plano.json: {"saida_wav": "voz.wav", "saida_falas": "falas.json", "saida_boca": "boca.json", "pausa_max": 0.09, "lufs": -14,
#              "segmentos": [{"arquivo": "...wav", "texto": "...", "voz": "clone|real", "pausa": 0.16, "ini": 0.12, "ganho_db": 0}]}
# falas.json = texto + tempo de cada palavra (legenda/animação); boca.json = dB por quadro a 30 fps (boca de personagem).
import json, sys, subprocess, numpy as np, difflib, unicodedata, re, sherpa_onnx
SR = 44100; FPS = 30
P = json.load(open(sys.argv[1]))
import os, tempfile
d = os.environ.get("VOZ_ASR_DIR", "/tmp/claude-0/asr/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8") + "/"
r = sherpa_onnx.OfflineRecognizer.from_transducer(encoder=d+"encoder.int8.onnx", decoder=d+"decoder.int8.onnx", joiner=d+"joiner.int8.onnx", tokens=d+"tokens.txt", num_threads=4, sample_rate=16000, feature_dim=128, model_type="nemo_transducer")
norm = lambda w: re.sub(r"[^a-z0-9]", "", unicodedata.normalize("NFKD", w.lower()).encode("ascii", "ignore").decode())
def le(f, ini=None, fim=None):
    a = ["ffmpeg", "-loglevel", "error"]
    if ini is not None: a += ["-ss", str(ini)]
    if fim is not None: a += ["-to", str(fim)]
    raw = subprocess.run(a + ["-i", f, "-ar", str(SR), "-ac", "1", "-f", "s16le", "-"], capture_output=True).stdout
    return np.frombuffer(raw, np.int16).astype(np.float32) / 32768
def apara(x, thr=-42):  # corta silêncio das pontas, deixa 40 ms; descarta um "estalo" curto antes da 1ª palavra
    h = 441; e = np.array([20 * np.log10(np.sqrt(np.mean(x[i:i+h]**2)) + 1e-9) for i in range(0, len(x) - h, h)])
    v = np.where(e > thr)[0]
    if not len(v): return x
    # regiões de som: se a primeira tem menos de 220 ms e vem seguida de >= 60 ms de silêncio, é o estalo do clone
    regs, ini = [], v[0]
    for a0, a1 in zip(v[:-1], v[1:]):
        if a1 - a0 > 1: regs.append((ini, a0)); ini = a1
    regs.append((ini, v[-1]))
    if len(regs) > 1 and (regs[0][1] - regs[0][0] + 1) * 10 < 220 and (regs[1][0] - regs[0][1]) * 10 >= 60:
        v = v[v >= regs[1][0]]
    a, b = max(0, (v[0] - 4) * h), min(len(x), (v[-1] + 5) * h)
    return x[a:b]
def encurta(x, alvo, thr=-40, minimo=0.12):
    """pausas internas maiores que `minimo` viram `alvo` segundos (corte no meio do silêncio, com 6 ms de rampa)"""
    h = 441; e = np.array([20 * np.log10(np.sqrt(np.mean(x[i:i+h]**2)) + 1e-9) for i in range(0, len(x) - h, h)])
    sil = e < thr; out = []; i = 0; ult = 0
    while i < len(sil):
        if sil[i]:
            j = i
            while j < len(sil) and sil[j]: j += 1
            if (j - i) * h / SR > minimo and i > 0 and j < len(sil):
                meio = (i + j) * h // 2; corta = int((j - i) * h - alvo * SR) // 2
                a, b = meio - corta, meio + corta
                seg = x[ult:a].copy(); r = int(0.006 * SR); seg[-r:] *= np.linspace(1, 0, r); out.append(seg)
                ult = b
            i = j
        else: i += 1
    resto = x[ult:].copy()
    if out: r = int(0.006 * SR); resto[:r] *= np.linspace(0, 1, r)
    out.append(resto)
    return np.concatenate(out)
def palavras_tempo(x, texto):
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes()
    raw = subprocess.run(["ffmpeg", "-loglevel", "error", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", "-", "-ar", "16000", "-f", "s16le", "-"], input=pcm, capture_output=True).stdout
    x16 = np.frombuffer(raw, np.int16).astype(np.float32) / 32768
    st = r.create_stream(); st.accept_waveform(16000, x16); r.decode_stream(st)
    asr = []
    for t, ts in zip(st.result.tokens, st.result.timestamps):
        if t.startswith(" ") or t.startswith("▁") or not asr: asr.append([t.strip(" ▁"), ts])
        else: asr[-1][0] += t
    mine = texto.split()
    sm = difflib.SequenceMatcher(None, [norm(w) for w in mine], [norm(a[0]) for a in asr], autojunk=False)
    ts = [None] * len(mine)
    for a, b, n in sm.get_matching_blocks():
        for k in range(n): ts[a + k] = asr[b + k][1]
    # interpola os que não casaram
    conh = [(i, t) for i, t in enumerate(ts) if t is not None]
    dur = len(x) / SR
    for i in range(len(ts)):
        if ts[i] is None:
            ant = max([c for c in conh if c[0] < i], default=(-1, 0.0), key=lambda c: c[0])
            pro = min([c for c in conh if c[0] > i], default=(len(ts), dur - 0.15), key=lambda c: c[0])
            ts[i] = ant[1] + (pro[1] - ant[1]) * (i - ant[0]) / (pro[0] - ant[0])
    return ts, st.result.text, len(conh)
voz, falas, t0 = [], [], 0.0
for s in P["segmentos"]:
    x = apara(le(s["arquivo"], s.get("ini"), s.get("fim")))
    x = encurta(x, P.get("pausa_max", 0.09))
    x = x * (10 ** (s.get("ganho_db", 0) / 20))
    ts, ouvido, casadas = palavras_tempo(x, s["texto"])
    print(f"{s['texto'][:50]!r:55} {len(x)/SR:5.2f}s casadas {casadas}/{len(s['texto'].split())} | ouvido: {ouvido}")
    falas.append({"texto": s["texto"], "ts": [round(t0 + t, 3) for t in ts], "voz": s.get("voz", "clone")})
    voz.append(x); t0 += len(x) / SR
    pausa = np.zeros(int(SR * s.get("pausa", 0.16)), np.float32); voz.append(pausa); t0 += len(pausa) / SR
y = np.concatenate(voz)
import wave
BRUTA = os.path.join(tempfile.gettempdir(), "voz-bruta.wav")
w = wave.open(BRUTA, "wb"); w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes((np.clip(y, -1, 1) * 32767).astype(np.int16).tobytes()); w.close()
subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", BRUTA, "-af", f"highpass=f=70,loudnorm=I={P.get('lufs', -14)}:TP=-2:LRA=7", "-ar", "44100", P["saida_wav"]], check=True)
y2 = le(P["saida_wav"]); q = SR // FPS
boca = [round(float(20 * np.log10(np.sqrt(np.mean(y2[i*q:(i+1)*q]**2)) + 1e-9)), 1) for i in range(len(y2) // q)]
json.dump(falas, open(P["saida_falas"], "w"), ensure_ascii=False, indent=1)
json.dump(boca, open(P["saida_boca"], "w"))
print("total", round(len(y2) / SR, 2), "s,", len(boca), "quadros")
