---
name: clonagem-voz
description: Clonar a voz do João (ou de qualquer pessoa com gravações próprias) para narração de reel, até ficar difícil de distinguir da voz real. Use quando for gerar narração com clone, quando a voz clonada soar robótica, lenta, com sotaque errado ou "não parecer ele", ou para escolher referência, motor e configuração de clonagem. Tudo é decidido por medida (semelhança de locutor, erro de texto, entonação e ritmo contra a voz real), nunca de ouvido.
---

# Clonagem de voz — da referência à narração montada

O clone não melhora com prompt nem com pós-processamento. Ele melhora com **referência certa** e
**escolha por medida**. Esta skill é o laboratório que provou isso e o caminho de produção que sai dele.

## O que foi medido (22/10/2026, voz do João, 3 frases do reel do PDF)

Régua: `sim` = semelhança de locutor (wespeaker) contra o centróide de 12 memos reais que não
entraram em nenhuma referência. Teto = um memo real contra os outros = **0,914**. Outra pessoa ≈ 0,11.

| configuração | sim | erro de texto | entonação (real 4,80) | ritmo pal/s (real 3,70) |
|---|---|---|---|---|
| antes: trecho da narração editada do Office, preset `broadcast`, 16 passos | 0,774 | 0,05 | 4,28 | 3,23 |
| mesmo trecho, `raw`, 32 passos | 0,767 | 0,08 | 4,39 | 3,57 |
| memo de 4,7 s | 0,848 | 0,10 | 4,38 | 3,63 |
| memo de 8,6 s, uma tomada | 0,900 | 0,09 | 5,38 | 3,34 |
| **dois memos, 16 s** | **0,910** | 0,08 | 4,85 | 3,21 |
| memo de 9,5 s, outro estilo (opinião, mais rápido) | 0,856 | 0,04 | 4,47 | 4,20 |
| memo de 12 s, tom monótono | 0,830 | 0,06 | 2,99 | 3,70 |
| Qwen3-TTS-0.6B-Base, melhor referência | 0,788 | 0,10 | 4,63 | 2,87 |

Rodada 3, em cima da vencedora (16 s):

| variação | sim | erro de texto | ritmo |
|---|---|---|---|
| seed 7 / 11 / 23 | 0,910 / 0,896 / 0,896 | 0,08 / 0,04 / 0,04 | ~3,2 |
| velocidade 1,12 | 0,912 | 0,12 | 3,46 |
| 16 s + memo rápido (18 s, estilos misturados) | 0,894 | 0,11 | 3,73 |
| **melhor take de 3 seeds, por fala** | **0,93 / 0,91 / 0,89** | ≤ 0,06 | — |

Acelerar troca palavra; misturar estilos piora identidade. O ritmo se acerta na montagem
(`monta.py` encurta as pausas internas), não no motor.

**Em produção (reel do Stirling-PDF, 7 falas, 31,9 s):** narração inteira foi de sim 0,787 para
**0,910** (teto real 0,914) e entonação de 4,16 para 4,75 st (real 4,80). Transcrição da narração
montada saiu sem nenhuma palavra trocada.

## As regras que saíram disso

1. **A referência é a alavanca. Mais nada chega perto.** Trocar só a referência levou de 0,77 a 0,91
   — o mesmo nível de uma gravação real dele contra as outras.
2. **Grave de verdade, não recorte narração editada.** Trecho com corte de edição, música ou efeito
   ensina o clone a soar editado. Use memo de uma tomada só, 8-20 s. Juntar dois memos do mesmo
   estilo com 0,3 s de silêncio funcionou (0,910) — o limite é 20 s no total (o VoiceStudio recusa
   referência maior com transcrição).
3. **O estilo da referência é o estilo do clone.** Memo monótono gera clone monótono (entonação 2,99);
   memo de opinião acelerado gera clone acelerado. Escolha a referência pelo estilo do que vai ser gerado:
   reel → memo falado com energia de reel.
4. **Transcrição exata, palavra por palavra.** `referencia.py` dá o rascunho do ASR; corrija à mão
   (o ASR escreveu "a Dub cau" onde ele disse "a Adobe caiu").
5. **`effect_preset=raw`.** O padrão do VoiceStudio é `broadcast` (compressor + EQ + limitador), que tira
   a cara de gravação real. Masterize só no fim, na narração montada.
6. **Nunca rode duas gerações ao mesmo tempo na CPU.** O VoiceStudio abandona job que passa de 600 s de
   CPU, e o job abandonado continua rodando e segurando a máquina. Em paralelo, tudo estoura.
7. **Gere vários takes e escolha por número.** O mesmo texto com seeds diferentes varia; `gera.py`
   fica com o take de maior semelhança entre os que não trocaram palavra.
8. **Take que trocou palavra não se aceita: reescreva a frase.** "clonar voz" virou "voice", "comprime"
   virou "com o Prime". Trocar a frase resolve; insistir no seed não.

## Motor

- **OmniVoice via VoiceStudio** (debpalash/voicestudio) é o melhor medido. **Licença do checkpoint:
  CC-BY-NC (não comercial)** — `model_licenses.json` marca `commercial_use: false`. Para reel orgânico
  é zona cinzenta; para anúncio pago ou produto vendido, não use.
- **Qwen3-TTS-12Hz-0.6B-Base** é Apache-2.0 (uso comercial liberado), roda ~5× o tempo real em 4 CPUs,
  mas perde em identidade (0,79 contra 0,91). É o plano B quando a licença importar.
- ElevenLabs não entra: é o inimigo pago dos vídeos.

## Caminho de produção

```
referencia.py "memos/*.m4a"                 # 1. lista candidatas (8-20 s, sinal/ruído, rascunho do texto)
referencia.py preparar memo.m4a r.wav       #    prepara a escolhida (pontas, 24 kHz mono, pico -6 dBFS)
audicao.py audicao.json                     # 2. mesmas 3 frases com cada referência/config
avalia.py audicao.json "memos/*.m4a" --fora id_ref1,id_ref2   # 3. ranking por nota
gera.py roteiro.json                        # 4. N takes por fala com a vencedora, escolhe por medida
estalo.py plano.json                        # 5. acha o corte do "É" que o clone solta antes da 1ª palavra
limpa.py plano.json                         # 6. monta (monta.py), transcreve tudo e corrige emendas
```

"REVISAR" no `gera.py` nem sempre é erro: confira o que o ASR ouviu. "para" no lugar de "pra" e
número em dígito já são normalizados; pronúncia aportuguesada de marca ("Acrobate") é do clone e
costuma passar — se incomodar, reescreva a frase.

Formatos dos JSON estão no cabeçalho de cada script. O `plano.json` do passo 5 é o `escolha.json` do
passo 4 mais `saida_wav`, `saida_falas`, `saida_boca` e as falas reais (`"voz": "real"`).

## Dependências

- VoiceStudio rodando: `OMNIVOICE_DEVICE=cpu uv run --no-sync python -m uvicorn backend.main:app --host 127.0.0.1 --port 3900`
  (cai de vez em quando; `audicao.py` espera ele voltar).
- `sherpa-onnx` com ASR Parakeet TDT 0.6B v3 int8 (`VOZ_ASR_DIR`) e os modelos de locutor wespeaker
  resnet34 e 3D-Speaker eres2net (`VOZ_SPK_DIR`). `ffmpeg`.
- Qwen (opcional): venv com `qwen-tts --no-deps` reaproveitando o torch do VoiceStudio por `PYTHONPATH`;
  `QWEN_MODELO` aponta para o checkpoint baixado.

## Transparência

Voz clonada em reel: diga que é clone quando for (o próprio reel do VoiceStudio faz disso o gancho).
Só clone a voz de quem gravou e autorizou.
