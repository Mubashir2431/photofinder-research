import glob
import onnxruntime as ort

cands = []
for p in glob.glob(r"models\zoo\antelopev2\**\*.onnx", recursive=True):
    try:
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        inp = s.get_inputs()[0].shape
        outs = [o.shape for o in s.get_outputs()]
        if inp == [1,3,112,112] and any((isinstance(o, list) and len(o)==2 and o[-1]==512) for o in outs):
            cands.append((p, inp, outs))
    except Exception:
        pass

print("CANDIDATES (112x112 -> 512):")
for p, inp, outs in cands:
    print(" -", p)
    print("    IN:", inp, "OUT:", outs)
