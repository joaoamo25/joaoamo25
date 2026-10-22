# uso: python3 estalo.py plano.json  (rode ANTES do monta.py)
# O clone solta um "É"/estalo de 0,1-0,2 s antes da 1ª palavra. Para cada fala do plano, acha o menor corte no início que faz a transcrição começar pela 1ª palavra do texto,
# testando com silêncio na frente (como a fala fica na montagem). Grava "ini" no plano.
import json, sys, subprocess, numpy as np, sherpa_onnx, unicodedata, re
import os
d=os.environ.get("VOZ_ASR_DIR","/tmp/claude-0/asr/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")+"/"
r=sherpa_onnx.OfflineRecognizer.from_transducer(encoder=d+"encoder.int8.onnx",decoder=d+"decoder.int8.onnx",joiner=d+"joiner.int8.onnx",tokens=d+"tokens.txt",num_threads=4,sample_rate=16000,feature_dim=128,model_type="nemo_transducer")
norm=lambda w: re.sub(r"[^a-z0-9]","",unicodedata.normalize("NFKD",w.lower()).encode("ascii","ignore").decode())
P=json.load(open(sys.argv[1]))
ant=np.zeros(16000,np.float32)
for s in P["segmentos"]:
    raw=subprocess.run(["ffmpeg","-loglevel","error","-i",s["arquivo"],"-ar","16000","-ac","1","-f","s16le","-"],capture_output=True).stdout
    x=np.frombuffer(raw,np.int16).astype(np.float32)/32768
    alvo=[norm(s["texto"].split()[0])]; escolha=None  # só a 1ª palavra: a 2ª esbarra em grafia do ASR (pra/para, Stirling-PDF)
    for c in [0,0.1,0.15,0.18,0.2,0.22,0.24,0.26,0.28,0.3]:
        oks=[]
        for ctx in (np.zeros(int(16000*0.15),np.float32), np.concatenate([ant[-16000:],np.zeros(int(16000*0.07),np.float32)])):
            y=np.concatenate([ctx,x[int(c*16000):]])
            st=r.create_stream(); st.accept_waveform(16000,y); r.decode_stream(st)
            toks=list(zip(st.result.tokens, st.result.timestamps)); t0=len(ctx)/16000-0.05
            txt="".join(t for t,ts in toks if ts>=t0); w=[norm(v) for v in txt.split()][:1]
            oks.append(w==alvo)
        if all(oks): escolha=c; break
    s["ini"]=escolha if escolha else 0
    ant=x[int(s["ini"]*16000):]
    print(s["arquivo"], "1ª palavra", alvo, "→ corte", escolha)
json.dump(P,open(sys.argv[1],"w"),ensure_ascii=False,indent=1)
