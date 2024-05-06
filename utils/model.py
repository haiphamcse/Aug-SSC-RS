from networks.dsc import DSC
# from networks.dsc_no_completion import DSC

def get_model(_cfg, phase='train'):
    return DSC(_cfg, phase=phase)
