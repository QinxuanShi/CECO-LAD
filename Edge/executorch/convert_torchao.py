import torch
import argparse
from torch.export import export, ExportedProgram
from executorch import exir
from executorch.exir import EdgeProgramManager, to_edge
from EMAT_model.EMAT import EMAT
from torchao.quantization import quantize_, int8_dynamic_activation_int4_weight

from torchao.utils import unwrap_tensor_subclass

def convert(config):
    precision = torch.float32
    device = torch.device("cpu")
    # initial EM-AT model
    model = EMAT(win_size=config.win_size, enc_in=config.input_c, c_out=config.output_c, e_layers=config.e_layer_num)
    
    params = 'e' + str(config.num_epochs) + '_k' + str(config.k) + '_l' + str(config.e_layer_num) + '_b' + str(config.batch_size)
    
    # this path is just an example of the .pth file path
    PATH = "./Cloud/checkpoints/ensemble_bgl/BGL_e3_k3_l3_b32.pth"

    model.load_state_dict(torch.load(PATH, map_location=device, weights_only=True))
    model.eval()
    model = model.to(dtype=precision, device="cpu")

    # only works for torch 2.4+

    # A8W4
    quantize_(model, int8_dynamic_activation_int4_weight())
    
    # torch lower than 2.6
    m_unwrapped = unwrap_tensor_subclass(model)
    
    sample_inputs = (torch.randn(1, config.win_size, config.data_seq_len, dtype=precision, device=device), )
    # lower to executorch and output to .pte file
    exported_program: ExportedProgram = export(m_unwrapped, sample_inputs)
    edge: EdgeProgramManager = to_edge(exported_program)
    executorch_program = edge.to_executorch()

    output_name = f"./{config.model_save_path}/{config.dataset}_{params}.pte"
    with open(output_name, "wb") as file:    
        file.write(executorch_program.buffer)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--k', type=int, default=3)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--e_layer_num', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--win_size', type=int, default=100)
    parser.add_argument('--data_seq_len', type=int, default=10) 
    parser.add_argument('--input_c', type=int, default=10)
    parser.add_argument('--output_c', type=int, default=10)
    parser.add_argument('--dataset', type=str, default='BGL')
    parser.add_argument('--model_save_path', type=str, default='checkpoints/qbat_bgl')

    config = parser.parse_args()

    args = vars(config)
    print('------------ Options -------------')
    for k, v in sorted(args.items()):
        print('%s: %s' % (str(k), str(v)))
    print('-------------- End ----------------')
    convert(config)
