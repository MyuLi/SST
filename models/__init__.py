from .SST import SST

def sst():
    net = SST(inp_channels=31,dim = 90,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_wdc():
    net = SST(inp_channels=191,dim = 210,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net

def sst_urban():
    net = SST(inp_channels=210,dim = 210,
        window_size=8,
        depths=[ 6,6,6,6,6,6],
        num_heads=[ 6,6,6,6,6,6],mlp_ratio=2)
    net.use_2dconv = True
    net.bandwise = False
    return net