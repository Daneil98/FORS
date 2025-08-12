import torch
print(torch.cuda.is_available())            # Should be True
print(torch.cuda.get_device_name(0))        # Should say "GeForce MX150"

print(f'{torch.version.cuda} \n')

import onnxruntime
print(onnxruntime.get_device())