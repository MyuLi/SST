from .SMSNet2 import SMSNet2

def smsnet2():
    net = SMSNet2(inp_channels=31,dim = 90,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net