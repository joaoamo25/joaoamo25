# Audição: gera as mesmas frases de teste com várias referências/configurações, para comparar com avalia.py.
# uso: python3 audicao.py audicao.json [cfg1,cfg2,...]
# audicao.json: {"saida": "aud", "linhas": ["frase 1", ...],
#                "cfgs": {"C1": {"ref": "r1.wav", "ref_text": "transcrição EXATA", "motor": "omnivoice",
#                                 "extra": {"num_step": 32, "speed": 1.0, "seed": 7}}}}
# motor "omnivoice" = VoiceStudio em http://127.0.0.1:3900 (VOICESTUDIO_URL); "qwen" = Qwen3-TTS-Base local (QWEN_MODELO).
# Gera um take por vez (CPU). Pula o que já existe. Nunca rode duas audições ao mesmo tempo: o VoiceStudio
# abandona job que passa de 600 s de CPU e o job abandonado continua segurando a máquina.
import json, os, sys, time, subprocess
P = json.load(open(sys.argv[1])); OUT = P.get("saida", "aud"); os.makedirs(OUT, exist_ok=True)
URL = os.environ.get("VOICESTUDIO_URL", "http://127.0.0.1:3900")
ordem = sys.argv[2].split(",") if len(sys.argv) > 2 else list(P["cfgs"])
_qwen = {}
def espera_backend():
    for _ in range(180):
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{http_code}", "-m", "3", URL + "/.well-known/voicestudio-speech"], capture_output=True, text=True).stdout
        if r == "200": return
        time.sleep(5)
    sys.exit("VoiceStudio não respondeu")
def omnivoice(c, texto, out):
    ex = {"num_step": 32, "guidance_scale": 2.0, "seed": 7, "effect_preset": "raw", **c.get("extra", {})}
    a = ["curl", "-s", "-X", "POST", URL + "/generate", "-F", f"text={texto}", "-F", "language=pt", "-F", f"ref_audio=@{c['ref']}",
         "-F", f"ref_text={c['ref_text']}", "-F", "engine=omnivoice", "--max-time", "900", "-o", out]
    for k, v in ex.items(): a[-4:-4] = ["-F", f"{k}={v}"]
    espera_backend(); subprocess.run(a)
def qwen(c, texto, out):
    import torch, soundfile as sf
    from qwen_tts import Qwen3TTSModel
    if "m" not in _qwen:
        torch.set_num_threads(os.cpu_count())
        _qwen["m"] = Qwen3TTSModel.from_pretrained(os.environ["QWEN_MODELO"], dtype=torch.float32, device_map="cpu")
    m = _qwen["m"]; chave = c["ref"]
    if chave not in _qwen: _qwen[chave] = m.create_voice_clone_prompt(ref_audio=c["ref"], ref_text=c["ref_text"])
    torch.manual_seed(c.get("extra", {}).get("seed", 7))
    w, sr = m.generate_voice_clone(text=texto, language="portuguese", voice_clone_prompt=_qwen[chave]); sf.write(out, w[0], sr)
for nome in ordem:
    c = P["cfgs"][nome]
    for i, t in enumerate(P["linhas"]):
        out = f"{OUT}/{nome}_a{i}.wav"
        if os.path.exists(out) and os.path.getsize(out) > 10000: continue
        t0 = time.time(); (qwen if c.get("motor") == "qwen" else omnivoice)(c, t, out)
        ok = os.path.getsize(out) > 10000
        print(json.dumps({"cfg": nome, "linha": i, "s": round(time.time() - t0, 1), "ok": ok, **({} if ok else {"erro": open(out, errors="ignore").read()[:300]})}, ensure_ascii=False), flush=True)
print("FIM", flush=True)
