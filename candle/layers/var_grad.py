from torch.autograd.function import Function
from torch import nn
import torch

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, var):
        ctx.save_for_backward(input,var)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # fowrward has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        #input , weight, bias = ctx.saved_tensors
        input, weight, bias, scaling_sign_grad, std_grad_noise = ctx.saved_tensors
        grad_input = grad_weight = grad_bias = None

        #print(torch.var(grad_output))
        #import pdb; pdb.set_trace()
        

        # These needs_input_grad checks are optional and there only to
        # improve efficiency. If you want to make your code simpler, you can
        # skip them. Returning gradients for inputs that don't require it is
        # not an error.
        if ctx.needs_input_grad[0]:
            grad_input = grad_output.mm(weight)
        #"""        
        #print(torch.std(grad_output))
        #import pdb; pdb.set_trace()
        #"""
        if ctx.needs_input_grad[1]:
            if scaling_sign_grad is not None:
                """
                J += alpha*mean*sign(J)
                or
                J += alpha*std*sign(J)
                or
                J += alpha*sign(J)
                
                grad_wight = J.t().mm(input)
                """
                #import pdb; pdb.set_trace()
                #grad_weight = scaling_sign_grad*torch.std(grad_output)*torch.sign(grad_output).t().mm(input)
                #grad_weight = scaling_sign_grad*torch.sign(grad_output).t().mm(input)
                grad_weight = scaling_sign_grad*torch.mean(grad_output)*torch.sign(grad_output).t().mm(input)

            elif std_grad_noise is not None:
                n = torch.randn_like(grad_output)
                grad_weight = (grad_output.add(std_grad_noise*n)).t().mm(input)
            else:
                grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)

        return grad_input, grad_weight, grad_bias, None,None


class VarGrad(nn.Module):
    def __init__(self):
        super(var_grad, self).__init__()
        self.var = nn.Parameter(torch.Tensor([0])

    def forward(self, input):
        return var_grad.apply(input, self.var)

