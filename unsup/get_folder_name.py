
###get the folder name for storing the result
def get_folder_name(args):
    folder_name = args.dataset+'_'+args.segmentation_model+'_'+str((args.max_iter)//1000)+'k_'
    if args.target_dataset == 'CrossCity': #target city
        folder_name += args.target_city + "_"
    folder_name += args.supervision_mode + "_"
    if args.supervision_mode == 'unsup': #unsup loss
        folder_name += str(args.unsup_loss)+"_"+str(args.uda_confidence_thresh)+'_'+str(args.unsup_coeff)
    elif args.supervision_mode == 'unsup_single': #unsup single loss
        folder_name += str(args.unsup_single_loss)+"_"+str(args.uda_confidence_thresh)+'_'+str(args.unsup_coeff)
    
    if args.loss_ohem: #use ohem
        folder_name += '_ohem_' + str(args.loss_ohem_ratio)
    if args.a_decay > 0: #use beta decay
            folder_name += '_decay_' + str(args.a_decay)
    if args.paste_mode != 'None': #paste mode
        folder_name += '_paste_mode_' + str(args.paste_mode) + '_%s'%str(args.mixed_weight)
    if args.aug_mix: #augment the mixed image
        folder_name += '_augmix'
    if args.class_relation: #classes relation
        folder_name += '_cls'
    if args.long_tail: #long tail class pasting
        folder_name += '_lt_' + str(args.long_tail_p)
    if args.bapa_boundary: #bapa boundary
        folder_name += '_bapa%i'%(args.bapa_boundary)
    if args.mixed_boundary: #boundary enhancement
        folder_name += '_mb%i'%(args.mixed_boundary)
        if args.mixed_gaussian_kernel > 0: #gaussian smoothing
            folder_name += '_gk%i'%(args.mixed_gaussian_kernel)
    if args.supervision_mode == 'sup': #supvised ratio of the dataset
        folder_name += ('_'+str(args.sup_ratio))
    if args.pretrained_path != 'None':
        print('args.pretrained_path', args.pretrained_path)
        folder_name += '_pre' 
    return folder_name