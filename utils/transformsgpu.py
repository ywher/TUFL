import torch

def flip(flip, data = None, target = None):
    #Flip
    if flip == 1:
        if not (data is None): data = torch.flip(data,(3,))
        if not (target is None):
            target = torch.flip(target,(2,))
    return data, target

def cowMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask, data = torch.broadcast_tensors(mask, data)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        data = (stackedMask*torch.cat((data[::2],data[::2]))+(1-stackedMask)*torch.cat((data[1::2],data[1::2]))).float()
    if not (target is None):
        stackedMask, target = torch.broadcast_tensors(mask, target)
        stackedMask = stackedMask.clone()
        stackedMask[1::2]=1-stackedMask[1::2]
        target = (stackedMask*torch.cat((target[::2],target[::2]))+(1-stackedMask)*torch.cat((target[1::2],target[1::2]))).float()
    return data, target

def mix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        if mask.shape[0] == data.shape[0]:
            data = torch.cat([(mask[i] * data[i] + (1 - mask[i]) * data[(i + 1) % data.shape[0]]).unsqueeze(0) for i in range(data.shape[0])])
        elif mask.shape[0] == data.shape[0] / 2:
            data = torch.cat((torch.cat([(mask[i] * data[2 * i] + (1 - mask[i]) * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))]),
                              torch.cat([((1 - mask[i]) * data[2 * i] + mask[i] * data[2 * i + 1]).unsqueeze(0) for i in range(int(data.shape[0] / 2))])))
    if not (target is None):
        target = torch.cat([(mask[i] * target[i] + (1 - mask[i]) * target[(i + 1) % target.shape[0]]).unsqueeze(0) for i in range(target.shape[0])])
    return data, target

def oneMix(mask, data = None, target = None):
    #Mix
    if not (data is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], data[0]) #(3,H,W) or (1,H,W)
        data = (stackedMask0*data[0]+(1-stackedMask0)*data[1]).unsqueeze(0) #(1,3,H,W)
    if not (target is None):
        stackedMask0, _ = torch.broadcast_tensors(mask[0], target[0])
        target = (stackedMask0*target[0]+(1-stackedMask0)*target[1]).unsqueeze(0)
    return data, target


def normalize(MEAN, STD, data = None, target = None):
    #Normalize
    if not (data is None):
        if data.shape[1]==3:
            STD = torch.Tensor(STD).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            MEAN = torch.Tensor(MEAN).unsqueeze(0).unsqueeze(2).unsqueeze(3).cuda()
            STD, data = torch.broadcast_tensors(STD, data)
            MEAN, data = torch.broadcast_tensors(MEAN, data)
            data = ((data-MEAN)/STD).float()
    return data, target
