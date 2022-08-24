import torch
import sys, os
root_folder = os.path.abspath(os.path.dirname(__file__) + os.path.sep + '..') #'/media/ywh/ubuntu/projects/BiSeNet-uda'
sys.path.append(root_folder)

def load_pretrained_model(args, net, pseudo_net, logger):
        if args.pretrained_dataset=='CityScapes':
            pretrained_path= os.path.join(root_folder, 'pretrained', 'cityscapes.pth')
        elif args.pretrained_dataset == 'CityScapes_Rome':
            pretrained_path= os.path.join(root_folder, 'pretrained', 'cityscapes_Rome.pth') 
        elif args.pretrained_dataset == 'CityScapes_Rio':
            pretrained_path= os.path.join(root_folder, 'pretrained', 'cityscapes_Rio.pth')
        elif args.pretrained_dataset == 'CityScapes_Tokyo':
            pretrained_path= os.path.join(root_folder, 'pretrained', 'cityscapes_Toyko.pth')
        elif args.pretrained_dataset == 'CityScapes_Taipei':
            pretrained_path= os.path.join(root_folder, 'pretrained', 'cityscapes_Taipei.pth') 
        elif args.pretrained_dataset=='Mapillary':
            pretrained_path= os.path.join(root_folder, 'pretrained', 'mapillary.pth') 
        #load pretrained supervised model
        if args.pretrained_path != 'None': #define the pretrained path
            pretrained_path = args.pretrained_path
        logger.info("pretrained_path "+str(pretrained_path))
        single_dict = torch.load(pretrained_path, map_location='cpu')
        #remove the last layer if the pretrained model has different classes which are the most usual case
        if args.pretrain_drop_last_layer:
            single_dict_final = single_dict.copy()
            for param_tensor in single_dict.keys():
                if param_tensor.count("conv_out") > 1:
                #if param_tensor.count("conv_out") > 1 or param_tensor.count("layer5"): #for bisenet and deeplab respectively
                    del single_dict_final[param_tensor]
            single_dict = single_dict_final
        net.load_state_dict(single_dict, strict=False)
        logger.info("pretrained_model loaded")
        if args.paste_mode != 'None':
            #load pretrained model
            pseudo_dict = torch.load(args.pesudo_pretrained_path, map_location='cpu')
            pseudo_net.load_state_dict(pseudo_dict, strict=True)
            logger.info("pretrained pseudo model loaded")
        return net, pseudo_net, logger