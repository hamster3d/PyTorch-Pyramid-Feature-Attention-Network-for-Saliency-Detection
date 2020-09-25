import argparse
import torch
from src.model import SODModel


def parse_arguments():
    parser = argparse.ArgumentParser(description='Parameters to convert model to ONNX.')
    parser.add_argument('--output', default='output.onnx', help='Path to onnx model', type=str)
    parser.add_argument('--model_path', default='best-model_epoch-204_mae-0.0505_loss-0.1370.pth', help='Path to model', type=str)
    parser.add_argument('--use_gpu', action="store_true", help='Whether to use GPU or not')
    parser.add_argument('--no_activation', action="store_true", help='Whether to use activation function before output')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()

    # Determine device
    if args.use_gpu and torch.cuda.is_available():
        device = torch.device(device='cuda')
    else:
        device = torch.device(device='cpu')

    # Load model
    model = SODModel(last_activation = not args.no_activation)
    chkpt = torch.load(args.model_path, map_location=device)
    model.load_state_dict(chkpt['model'])
    model.to(device)
    model.eval()

    batch_size = 1
    x = torch.randn(batch_size, 3, 256, 256, requires_grad=True)
    torch_out = model(x)

    torch.onnx.export(
        model,              # model being run
        x,                  # model input (or a tuple for multiple inputs)
        args.output,        # where to save the model (can be a file or file-like object)
        export_params=True,        # store the trained parameter weights inside the model file
        opset_version=11,          # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names = ['input'],   # the model's input names
        output_names = ['output'], # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                      'output' : {0 : 'batch_size'}})