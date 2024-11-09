import os
import torch
from torch.autograd import Variable
from net import NestFuse_light2_nodense, Fusion_network, Fusion_strategy, RFN_decoder
import utils
from args_fusion import args
import numpy as np


def load_model(path_auto, path_fusion, fs_type, flag_img):
    # Set number of input channels, if flag_img is True, it implies RGB images
    if flag_img is True:
        nc = 3
    else:
        nc = 1
    input_nc = nc
    output_nc = nc
    nb_filter = [64, 112, 160, 208]

    nest_model = NestFuse_light2_nodense(nb_filter, input_nc, output_nc, deepsupervision=False)
    nest_model.load_state_dict(torch.load(path_auto))  # Path for the autoencoder model

    fusion_model = Fusion_network(nb_filter, fs_type)
    fusion_model.load_state_dict(torch.load(path_fusion))  # Path for the fusion model

    fusion_strategy = Fusion_strategy(fs_type)

    para = sum([np.prod(list(p.size())) for p in nest_model.parameters()])
    print('Model {} : params: {:4f}M'.format(nest_model._get_name(), para * 4 / 1000 / 1000))

    para = sum([np.prod(list(p.size())) for p in fusion_model.parameters()])
    print('Model {} : params: {:4f}M'.format(fusion_model._get_name(), para * 4 / 1000 / 1000))

    nest_model.eval()
    fusion_model.eval()

    nest_model.cuda()
    fusion_model.cuda()

    return nest_model, fusion_model, fusion_strategy


def run_demo(nest_model, fusion_model, fusion_strategy, infrared_path, visible_path, output_path_root, name_ir, fs_type,
             use_strategy, flag_img, alpha):
    print(f"Processing pair: IR image - {infrared_path}, VIS image - {visible_path}")
    img_ir, h, w, c = utils.get_test_image(infrared_path, flag=flag_img)
    img_vi, h, w, c = utils.get_test_image(visible_path, flag=flag_img)

    if c == 1:
        if args.cuda:
            img_ir = img_ir.cuda()
            img_vi = img_vi.cuda()
        img_ir = Variable(img_ir, requires_grad=False)
        img_vi = Variable(img_vi, requires_grad=False)
        en_r = nest_model.encoder(img_ir)
        en_v = nest_model.encoder(img_vi)
        f = fusion_model(en_r, en_v)
        img_fusion_list = nest_model.decoder_eval(f)
    else:
        img_fusion_blocks = []
        for i in range(c):
            img_vi_temp = img_vi[i]
            img_ir_temp = img_ir[i]
            if args.cuda:
                img_vi_temp = img_vi_temp.cuda()
                img_ir_temp = img_ir_temp.cuda()
            img_vi_temp = Variable(img_vi_temp, requires_grad=False)
            img_ir_temp = Variable(img_ir_temp, requires_grad=False)

            en_r = nest_model.encoder(img_ir_temp)
            en_v = nest_model.encoder(img_vi_temp)
            f = fusion_model(en_r, en_v)
            img_fusion_temp = nest_model.decoder_eval(f)
            img_fusion_blocks.append(img_fusion_temp)
        img_fusion_list = utils.recons_fusion_images(img_fusion_blocks, h, w)

    output_count = 0
    for img_fusion in img_fusion_list:
        file_name = 'fused_' + alpha + '_' + name_ir
        output_path = output_path_root + file_name
        output_count += 1
        utils.save_image_test(img_fusion, output_path)
        print(output_path)


def main():
    flag_img = False
    ir_path = "<path_to_infrared_images_directory>"  # Placeholder for IR images directory path
    vis_path = "<path_to_visible_images_directory>"  # Placeholder for VIS images directory path
    path_auto = args.resume_nestfuse
    output_path_root = "<output_directory_path>"  # Placeholder for the output directory path

    print(f"Current working directory: {os.getcwd()}")
    print(f"IR directory exists: {os.path.exists(ir_path)}")
    print(f"VIS directory exists: {os.path.exists(vis_path)}")

    if os.path.exists(output_path_root) is False:
        os.mkdir(output_path_root)

    fs_type = 'res'  # Options: 'res' (RFN), 'add', 'avg', 'max', 'spa', 'nuclear'
    use_strategy = False  # True for static strategy, False for RFN

    path_fusion_root = "<path_to_fusion_model_directory>"  # Placeholder for fusion model directory path

    with torch.no_grad():
        alpha_list = [700]
        w_all_list = [[6.0, 3.0]]

        for alpha in alpha_list:
            for w_all in w_all_list:
                w, w2 = w_all

                temp = 'rfnnest_' + str(alpha) + '_wir_' + str(w) + '_wvi_' + str(w2) + 'axial'
                output_path_list = 'fused_' + temp + '_21' + '_' + fs_type
                output_path1 = output_path_root + output_path_list + '/'
                if os.path.exists(output_path1) is False:
                    os.mkdir(output_path1)
                output_path = output_path1

                path_fusion = "<path_to_specific_fusion_model>"  # Placeholder for a specific fusion model path

                model, fusion_model, fusion_strategy = load_model(path_auto, path_fusion, fs_type, flag_img)

                imgs_paths_ir = sorted([os.path.join(ir_path, f) for f in os.listdir(ir_path) if
                                        (f.endswith(".jpg") or f.endswith(".png")) and "testIR_" in f])
                num = len(imgs_paths_ir)
                print(f"Found {num} IR images for testing.")

                for i in range(num):
                    infrared_path = imgs_paths_ir[i]
                    name_ir = os.path.basename(infrared_path)
                    name_vis = name_ir.replace('IR', 'VIS').replace('.png', '.jpg')
                    visible_path = os.path.join(vis_path, name_vis)

                    if not os.path.exists(visible_path):
                        name_vis = name_ir.replace('IR', 'VIS').replace('.jpg', '.png')
                        visible_path = os.path.join(vis_path, name_vis)

                    if not os.path.exists(visible_path):
                        print(f"Warning: VIS image not found for {infrared_path}")
                        continue

                    print(f"Processing IR image: {infrared_path}, corresponding VIS image: {visible_path}")

                    run_demo(model, fusion_model, fusion_strategy, infrared_path, visible_path, output_path, name_ir,
                             fs_type, use_strategy, flag_img, temp)

                print('Done......')


if __name__ == '__main__':
    main()
