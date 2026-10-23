import unittest

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper

from tinyocr.compress import convert, quantize_channel


def graph(weight, op="MatMul", **attributes):
    inp=helper.make_tensor_value_info('x',TensorProto.FLOAT,[1,16])
    out=helper.make_tensor_value_info('y',TensorProto.FLOAT,[1,16])
    model=helper.make_model(helper.make_graph([helper.make_node(op,['x','w'],['y'],**attributes)],
                  'test',[inp],[out],[numpy_helper.from_array(weight,'w')]),opset_imports=[helper.make_opsetid('',11)])
    model.ir_version=10
    helper.set_model_props(model,{'character':'a\nb\nc'})
    return model


class CompressionTests(unittest.TestCase):
    def test_quantization_known_values_and_zero_channel(self):
        w=np.array([[0,1,-1],[0,.5,-.5]],dtype=np.float32)
        q,s=quantize_channel(w,1)
        np.testing.assert_array_equal(q,[[0,127,-127],[0,64,-64]])
        self.assertEqual(float(s[0,0]),1)
        self.assertTrue(np.isfinite(q.astype(np.float32)*s).all())
        self.assertTrue(np.all(np.abs(w-q*s)<=s/2+1e-7))

    def test_onnx_executes_decompressed_weights_and_preserves_metadata(self):
        w=np.linspace(-1,1,256,dtype=np.float32).reshape(16,16)
        source=graph(w);before=source.SerializeToString()
        candidate,records=convert(source,'w8')
        self.assertEqual(source.SerializeToString(),before)
        self.assertEqual(len(records),1)
        self.assertEqual(candidate.metadata_props[0].value,'a\nb\nc')
        options=ort.SessionOptions();options.intra_op_num_threads=options.inter_op_num_threads=1
        s=ort.InferenceSession(candidate.SerializeToString(),options,providers=['CPUExecutionProvider'])
        x=np.arange(16,dtype=np.float32).reshape(1,16)/16
        q,scale=quantize_channel(w,1)
        np.testing.assert_allclose(s.run(None,{'x':x})[0],x@(q.astype(np.float32)*scale),rtol=1e-6,atol=1e-6)

    def test_float16_storage_cast(self):
        model,records=convert(graph(np.full((16,16),1.0001,dtype=np.float32)),'w16')
        stored=next(t for t in model.graph.initializer if t.data_type==TensorProto.FLOAT16)
        np.testing.assert_array_equal(numpy_helper.to_array(stored),np.ones((16,16),dtype=np.float16))
        self.assertEqual(records[0]['elements'],256)
        onnx.checker.check_model(model,full_check=True)

    def test_gemm_transpose_axis(self):
        _,records=convert(graph(np.ones((16,16),dtype=np.float32),'Gemm',transB=1),'w8')
        self.assertEqual(records[0]['axis'],0)

    def test_nonfinite_rejected(self):
        w=np.ones((16,16),dtype=np.float32);w[0,0]=np.nan
        with self.assertRaises(ValueError):convert(graph(w),'w8')

    def test_unsupported_consumer_untouched(self):
        model=graph(np.ones((16,16),dtype=np.float32))
        model.graph.node[0].op_type='Add'
        # Add broadcasts [1,16] and [16,16] to [16,16], unlike MatMul.
        model.graph.output[0].type.tensor_type.shape.dim[0].dim_value=16
        _,records=convert(model,'w8')
        self.assertEqual(records,[])
