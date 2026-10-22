# Passo 1: acha as melhores referências entre as gravações reais e prepara o .wav.
# uso: python3 referencia.py "pasta/*.m4a"                 → lista candidatas (duração, pico, ruído, transcrição do ASR)
#      python3 referencia.py preparar entrada.m4a saida.wav → corta silêncio das pontas, mono 24 kHz, pico em -6 dBFS
#      python3 referencia.py juntar a.wav b.wav saida.wav   → une referências preparadas com 0,3 s de silêncio (total ≤ 20 s)
# Regra medida: referência de UMA tomada só, 8-20 s, fala no MESMO estilo do que vai ser gerado.
# A transcrição do ASR é rascunho: corrija à mão palavra por palavra antes de usar (ref_text errado = clone pior).
import sys, glob, os, subprocess, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from voz_lib import *
if sys.argv[1] == "preparar":
    ent, sai = sys.argv[2], sys.argv[3]
    f = "silenceremove=start_periods=1:start_threshold=-45dB,areverse,silenceremove=start_periods=1:start_threshold=-45dB,areverse"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", ent, "-af", f, "-ar", "24000", "-ac", "1", sai], check=True)
    x = le16(sai); pico = 20 * np.log10(np.abs(x).max() + 1e-9)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", sai, "-af", f"volume={-6 - pico:.2f}dB", sai + ".tmp.wav"], check=True); os.replace(sai + ".tmp.wav", sai)
    d = len(le16(sai)) / 16000; print(f"{sai}: {d:.1f} s" + ("  ATENÇÃO: fora de 8-20 s" if not 8 <= d <= 20 else "")); sys.exit()
if sys.argv[1] == "juntar":
    *ents, sai = sys.argv[2:]; n = len(ents)
    ins = sum([["-i", e] for e in ents], []) + ["-f", "lavfi", "-t", "0.3", "-i", "anullsrc=r=24000:cl=mono"]
    seq = "".join(f"[{i}][{n}]" for i in range(n - 1)) + f"[{n - 1}]"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", *ins, "-filter_complex", f"{seq}concat=n={2 * n - 1}:v=0:a=1", "-ar", "24000", "-ac", "1", sai], check=True)
    d = len(le16(sai)) / 16000; print(f"{sai}: {d:.1f} s" + ("  PASSOU de 20 s: o VoiceStudio recusa" if d > 20 else "")); sys.exit()
linhas = []
for f in sorted(glob.glob(sys.argv[1])):
    x = le16(f); d = len(x) / 16000
    if d < 3: continue
    q = np.array([np.sqrt(np.mean(x[i:i+320]**2)) + 1e-9 for i in range(0, len(x) - 320, 320)])
    snr = 20 * np.log10(np.percentile(q, 95) / np.percentile(q, 10))
    linhas.append((d, f, 20 * np.log10(np.abs(x).max() + 1e-9), snr, asr(x)))
for d, f, p, snr, t in sorted(linhas, reverse=True):
    tag = "BOA" if 8 <= d <= 20 and snr > 25 else ("curta" if d < 8 else ("longa" if d > 20 else "ruído"))
    print(f"{tag:5} {d:5.1f}s pico {p:5.1f} dB  sinal/ruído {snr:4.0f} dB  {os.path.basename(f)}\n      {t}")
