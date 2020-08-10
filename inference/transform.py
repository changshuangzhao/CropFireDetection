import os
import sys
import torch
import onnx
from onnx_tf.backend import prepare
from torchsummary import summary

sys.path.append(os.path.join(os.path.dirname(__file__), '../src/networks'))
from mobilenet import mobilenet_v2


def pth2onnx(pth_path):
    net = mobilenet_v2(pretrained=False, num_classes=2).to('cpu')
    checkpoint = torch.load(pth_path, map_location=lambda storage, loc: storage)
    net.load_state_dict(checkpoint['state_dict'])
    # summary(net, (3, 112, 112))
    net.eval()

    dummy_input = torch.tensor(torch.randn(1, 3, 112, 112))
    export_onnx_file = 'models/model.onnx'
    torch.onnx.export(net, dummy_input, export_onnx_file,
                      opset_version=10,
                      do_constant_folding=True,
                      input_names=['img_input'],
                      output_names=['cls_out'],
                      # dynamic_axes={'images': {0: 'batch_size'},
                      #               'location': {0: 'batch_szie'},
                      #               'confidence': {0: 'batch_szie'}}
    )


def onnx2pb(onnx_path):
    model = onnx.load(onnx_path)
    tf_rep = prepare(model)
    print('inputs: ', tf_rep.inputs)
    print('outputs: ', tf_rep.outputs)
    tf_rep.export_graph('models/model.pb')


if __name__ == '__main__':
    pth_path = os.path.join(os.path.dirname(__file__), '../src/train/weights/98.170model_best.pth')
    pth2onnx(pth_path)
    onnx_path = os.path.join(os.path.dirname(__file__), 'models/model.onnx')
    onnx2pb(onnx_path)