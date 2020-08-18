from torch.autograd.function import Function
from torch import nn
import torch

class LinearFunction(Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None, scaling_sign_grad=None ,std_grad_noise=None):
        ctx.save_for_backward(input, weight,bias, scaling_sign_grad, std_grad_noise)
        output = input.mm(weight.t())
        if bias is not None:
            output += bias.unsqueeze(0).expand_as(output)
        return output

    # fowrward has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):
        ###  Output
        # ## ----------------------------------------------
        ###

        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        #input , weight, bias = ctx.saved_tensors
        input, weight, bias, scaling_sign_grad, std_grad_noise= ctx.saved_tensors
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
                Original type:
                    g += alpha*sign(g)
                """                
                #grad_weight = scaling_sign_grad*torch.sign(grad_output).t().mm(input)

                """
                MEAN-type:
                    g += alpha*mean*sign(g)
                """
                grad_weight = scaling_sign_grad*torch.mean(grad_output)*torch.sign(grad_output).t().mm(input)

                """
                STD-type:
                    g += alpha*std*sign(g)
                """
                #grad_weight = scaling_sign_grad*torch.std(grad_output)*torch.sign(grad_output).t().mm(input)  
            elif std_grad_noise is not None:
                n = torch.randn_like(grad_output)
                grad_weight = (grad_output.add(std_grad_noise*n)).t().mm(input)
            else:
                grad_weight = grad_output.t().mm(input)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0).squeeze(0)


        return grad_input, grad_weight, grad_bias, None,None


class Linear(nn.Module):
    def __init__(self, input_features, output_features, bias=False, std_grad_noise=None, scaling_sign_grad=None):
        super(Linear, self).__init__()
        self.input_features = input_features
        self.output_features = output_features
        # nn.Parameter is a special kind of Tensor, that will get
        # automatically registered as Module's parameter once it's assigned
        # as an attribute. Parameters and buffers need to be registered, or
        # they won't appear in .parameters() (doesn't apply to buffers), and
        # won't be converted when e.g. .cuda() is called. You can use
        # .register_buffer() to register buffers.
        # nn.Parameters require gradients by default.
        self.weight = nn.Parameter(torch.Tensor(output_features, input_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(output_features))
        else:
            # You should always register all possible parameters, but the
            # optional ones can be None if you want.
            self.register_parameter('bias', None)

        if std_grad_noise:
            self.std_grad_noise = nn.Parameter(torch.Tensor([std_grad_noise]), requires_grad=False)
        else:
            self.register_parameter('std_grad_noise', None)

        if scaling_sign_grad:            
            self.scaling_sign_grad = nn.Parameter(torch.Tensor([scaling_sign_grad]), requires_grad=False)
        else:
            self.register_parameter("scaling_sign_grad", None)



    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        return LinearFunction.apply(input, self.weight, self.bias,  \
            self.scaling_sign_grad, self.std_grad_noise)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )


