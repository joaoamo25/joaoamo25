# Régua da clonagem: identidade (embedding de locutor), fidelidade ao texto (ASR) e ritmo/entonação.
import subprocess, numpy as np, sherpa_onnx, unicodedata, re, difflib
import os
SPK_DIR = os.environ.get("VOZ_SPK_DIR", "/tmp/claude-0/spk")
ASR_DIR = os.environ.get("VOZ_ASR_DIR", "/tmp/claude-0/asr/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8")
SPK = {"wespeaker": SPK_DIR + "/wespeaker_en_voxceleb_resnet34.onnx", "eres2net": SPK_DIR + "/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx"}
_ext = {}
def le16(f, ini=None, fim=None):
    a = ["ffmpeg", "-loglevel", "error"] + (["-ss", str(ini)] if ini is not None else []) + (["-to", str(fim)] if fim is not None else []) + ["-i", f, "-ar", "16000", "-ac", "1", "-f", "s16le", "-"]
    return np.frombuffer(subprocess.run(a, capture_output=True).stdout, np.int16).astype(np.float32) / 32768
def emb(x, modelo="wespeaker"):
    if modelo not in _ext:
        _ext[modelo] = sherpa_onnx.SpeakerEmbeddingExtractor(sherpa_onnx.SpeakerEmbeddingExtractorConfig(model=SPK[modelo], num_threads=4))
    e = _ext[modelo]; st = e.create_stream(); st.accept_waveform(16000, x); st.input_finished()
    v = np.array(e.compute(st)); return v / np.linalg.norm(v)
_asr = None
def asr(x):
    global _asr
    if _asr is None:
        d = ASR_DIR + "/"
        _asr = sherpa_onnx.OfflineRecognizer.from_transducer(encoder=d+"encoder.int8.onnx", decoder=d+"decoder.int8.onnx", joiner=d+"joiner.int8.onnx", tokens=d+"tokens.txt", num_threads=4, sample_rate=16000, feature_dim=128, model_type="nemo_transducer")
    st = _asr.create_stream(); st.accept_waveform(16000, x); _asr.decode_stream(st); return st.result.text
_U = "zero um dois tres quatro cinco seis sete oito nove dez onze doze treze quatorze quinze dezesseis dezessete dezoito dezenove".split()
_D = "_ _ vinte trinta quarenta cinquenta sessenta setenta oitenta noventa".split()
_C = "_ cento duzentos trezentos quatrocentos quinhentos seiscentos setecentos oitocentos novecentos".split()
def _extenso(n):
    """inteiro 0..999999 por extenso (pt-BR, sem acento) — o ASR escreve dígitos, o roteiro escreve por extenso"""
    if n < 20: return _U[n]
    if n < 100: return _D[n // 10] + ("" if n % 10 == 0 else " e " + _U[n % 10])
    if n == 100: return "cem"
    if n < 1000: return _C[n // 100] + ("" if n % 100 == 0 else " e " + _extenso(n % 100))
    m, r = divmod(n, 1000); return ("mil" if m == 1 else _extenso(m) + " mil") + ("" if r == 0 else " e " + _extenso(r))
def norm(t):
    t = unicodedata.normalize("NFKD", t.lower()).encode("ascii", "ignore").decode()
    t = re.sub(r"r\$ ?(\d+)(,00)?", r"\1 reais", t); t = t.replace("%", " por cento")
    t = re.sub(r"\d+", lambda m: " " + _extenso(int(m.group())) + " " if int(m.group()) < 10**6 else m.group(), t)
    t = re.sub(r"\bmb\b", "mega", t); t = re.sub(r"\bpra\b", "para", t); t = re.sub(r"\bpro\b", "para o", t)
    return re.sub(r"[^a-z0-9 ]", " ", t).split()
def wer(ref, hip):
    a, b = norm(ref), norm(hip); sm = difflib.SequenceMatcher(None, a, b)
    return 1 - sum(m.size for m in sm.get_matching_blocks()) / max(1, len(a))
def f0(x):
    """F0 por autocorrelação simples (10 ms), só nos quadros sonoros"""
    h, w = 160, 640; out = []
    for i in range(0, len(x) - w, h):
        s = x[i:i+w]
        if np.sqrt(np.mean(s**2)) < 0.01: continue
        s = s - s.mean(); ac = np.correlate(s, s, "full")[w-1:]
        lo, hi = 16000 // 350, 16000 // 70
        k = lo + np.argmax(ac[lo:hi]);
        if ac[k] > 0.3 * ac[0]: out.append(16000 / k)
    return np.array(out)
def prosodia(x):
    p = f0(x); st = 12 * np.log2(p / np.median(p)) if len(p) > 10 else np.array([0])
    return {"f0_med": round(float(np.median(p)), 1) if len(p) else 0, "f0_var_st": round(float(np.std(st)), 2)}
