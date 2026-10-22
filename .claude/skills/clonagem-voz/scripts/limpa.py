# uso: python3 limpa.py plano.json  (substitui o monta.py como passo final)
# Monta a narração, transcreve tudo e, em cada emenda, vê se apareceu palavra a mais antes da 1ª palavra da fala.
# Se apareceu, corta mais 40 ms do início daquela fala e repete (até 8 rodadas).
import json, sys, subprocess, numpy as np, sherpa_onnx, unicodedata, re
import os
d=os.environ.get("VOZ_ASR_DIR","/tmp/claude-0/asr/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")+"/"
r=sherpa_onnx.OfflineRecognizer.from_transducer(encoder=d+"encoder.int8.onnx",decoder=d+"decoder.int8.onnx",joiner=d+"joiner.int8.onnx",tokens=d+"tokens.txt",num_threads=4,sample_rate=16000,feature_dim=128,model_type="nemo_transducer")
norm=lambda w: re.sub(r"[^a-z0-9]","",unicodedata.normalize("NFKD",w.lower()).encode("ascii","ignore").decode())
plano=sys.argv[1]; P=json.load(open(plano)); PUB=P["saida_wav"]; DIC=P["saida_falas"]
MONTA=os.path.join(os.path.dirname(os.path.abspath(__file__)),"monta.py")
for rodada in range(8):
    subprocess.run(["python3",MONTA,plano],capture_output=True,check=True)
    raw=subprocess.run(["ffmpeg","-loglevel","error","-i",PUB,"-ar","16000","-ac","1","-f","s16le","-"],capture_output=True).stdout
    x=np.frombuffer(raw,np.int16).astype(np.float32)/32768
    st=r.create_stream(); st.accept_waveform(16000,x); r.decode_stream(st)
    words=[]  # (palavra, tempo)
    for t,ts in zip(st.result.tokens, st.result.timestamps):
        if t.startswith(" ") or not words: words.append([t.strip(), ts])
        else: words[-1][0]+=t
    F=json.load(open(DIC)); mexeu=False
    for k in range(1,len(F)):
        ini=F[k]["ts"][0]; prev_fim=F[k-1]["ts"][-1]
        # palavras ouvidas entre a última palavra da fala anterior (+0,25 s) e a 1ª desta (−0,05 s)
        p1=norm(F[k]["texto"].split()[0])
        # a própria 1ª palavra pode cair antes do tempo alinhado: isso não é estalo
        extra=[w for w,ts in words if prev_fim+0.35 < ts < ini-0.02 and norm(w) and norm(w)!=p1]
        if extra:
            s=P["segmentos"][k]; s["ini"]=round(s.get("ini",0)+0.04,3); mexeu=True
            print(f"rodada {rodada}: fala {k+1} tinha {extra} antes de '{F[k]['texto'].split()[0]}' → corte {s['ini']}")
    json.dump(P,open(plano,"w"),ensure_ascii=False,indent=1)
    if not mexeu: print("limpo na rodada", rodada); break
print(st.result.text)
