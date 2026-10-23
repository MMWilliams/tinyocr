"""Weight storage compression with float32 compute; no calibration or training."""
import argparse
import copy
import hashlib
import shutil

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from .common import ROOT, sha256, write_json

SOURCE_HASHES = {
    "PP-OCRv6_det_tiny.onnx": "f42c0fbd294d95eac1a550e131b277dac97462c8025fa4b6c3cec1b7894bd3d5",
    "PP-OCRv6_rec_tiny.onnx": "e16e242de5937ad92609223f19bc2aff3727ee40b095f996907c24749bad251b",
}


def quantize_channel(weight, axis):
    if weight.dtype != np.float32 or not np.isfinite(weight).all():
        raise ValueError("Expected finite float32 weights")
    reduction = tuple(i for i in range(weight.ndim) if i != axis)
    maximum = np.max(np.abs(weight), axis=reduction, keepdims=True)
    scale = np.where(maximum == 0, 1, np.maximum(maximum/127, np.finfo(np.float32).tiny)).astype(np.float32)
    quantized = np.rint(weight/scale).clip(-127,127).astype(np.int8)
    return quantized, scale


def convert(model, mode):
    if mode not in ("w8", "w16"):
        raise ValueError("Unknown compression mode")
    model = copy.deepcopy(model)
    users = {}
    names = {i.name for i in model.graph.initializer}
    for node in model.graph.node:
        names.update(node.input); names.update(node.output)
        for index,name in enumerate(node.input):
            users.setdefault(name,[]).append((node,index))
    initializers, prefix_nodes, records = [], [], []
    for tensor in model.graph.initializer:
        weight = numpy_helper.to_array(tensor)
        consumers = users.get(tensor.name,[])
        axes = []
        for node,index in consumers:
            if index != 1:
                axes.append(None)
            elif node.op_type == "Conv" and weight.ndim >= 3:
                axes.append(0)
            elif node.op_type == "MatMul" and weight.ndim == 2:
                axes.append(1)
            elif node.op_type == "Gemm" and weight.ndim == 2:
                trans_b = next((a.i for a in node.attribute if a.name == "transB"),0)
                axes.append(0 if trans_b else 1)
            else:
                axes.append(None)
        if weight.dtype != np.float32 or weight.size < 256 or not axes or None in axes or len(set(axes)) != 1:
            initializers.append(tensor)
            continue
        if not np.isfinite(weight).all():
            raise ValueError("Nonfinite weight")
        prefix = "tinyocr_"+hashlib.sha256(tensor.name.encode()).hexdigest()[:16]
        generated = [prefix+s for s in ("_stored","_cast","_scale")]
        if any(n in names for n in generated):
            raise ValueError("Generated graph name collides")
        names.update(generated)
        stored_name,cast_name,scale_name = generated
        if mode == "w8":
            stored,scale = quantize_channel(weight,axes[0])
            restored = stored.astype(np.float32)*scale
            initializers.extend([numpy_helper.from_array(stored,stored_name),numpy_helper.from_array(scale,scale_name)])
            prefix_nodes.extend([helper.make_node("Cast",[stored_name],[cast_name],to=TensorProto.FLOAT),
                                 helper.make_node("Mul",[cast_name,scale_name],[tensor.name])])
        else:
            stored = weight.astype(np.float16)
            if not np.isfinite(stored).all():
                raise ValueError("Weight overflows float16")
            restored = stored.astype(np.float32)
            initializers.append(numpy_helper.from_array(stored,stored_name))
            prefix_nodes.append(helper.make_node("Cast",[stored_name],[tensor.name],to=TensorProto.FLOAT))
        records.append({"name":tensor.name,"shape":list(weight.shape),"axis":axes[0],
                        "elements":weight.size,"max_abs_error":float(np.max(np.abs(weight-restored)))})
    del model.graph.initializer[:]; model.graph.initializer.extend(initializers)
    nodes = list(model.graph.node)
    del model.graph.node[:]; model.graph.node.extend(prefix_nodes+nodes)
    onnx.checker.check_model(model,full_check=True)
    return model,records


def export(mode):
    directory = ROOT/"models"/f"v6-{mode}"
    directory.mkdir(parents=True,exist_ok=True)
    artifacts=[]
    for filename,expected in SOURCE_HASHES.items():
        source=ROOT/"models/v6-tiny"/filename
        if sha256(source) != expected:
            raise ValueError("Source baseline hash differs from frozen reference")
        destination=directory/filename
        actual_mode = ('original' if 'det_' in filename else 'w8') if mode=='w8-rec' else mode
        if actual_mode=='original':
            records=[]
            shutil.copyfile(source,destination)
        else:
            model,records=convert(onnx.load(source),actual_mode)
            onnx.save(model,destination)
        artifacts.append({"path":destination.relative_to(ROOT).as_posix(),"bytes":destination.stat().st_size,
                          "sha256":sha256(destination),"source_sha256":expected,"weights":records})
    report={"model":f"v6-{mode}","onnx_version":onnx.__version__,"mode":mode,
            "method":{"w8":"int8 per output channel + Cast/Mul","w16":"float16 weight initializers + Cast",
                      "w8-rec":"original float32 detector + int8 per-channel recognizer"}[mode],
            "compute":"float32; constant weight decoding can fold at load; no integer kernel or RAM claim",
            "calibration":"none","training":"none","artifacts":artifacts,
            "artifact_bytes":sum(a["bytes"] for a in artifacts)}
    write_json(ROOT/f"results/export-v6-{mode}.json",report)
    print(f"v6-{mode}: {report['artifact_bytes']:,} bytes; {sum(len(a['weights']) for a in artifacts)} weight tensors")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=["w8","w16","w8-rec"],default="w8")
    export(parser.parse_args().mode)
