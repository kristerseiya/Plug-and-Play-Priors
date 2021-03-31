
import torch
from . import model as net

def single2tensor4(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().unsqueeze(0)

def tensor2single(img):
    img = img.data.squeeze().float().cpu().numpy()
    if img.ndim == 3:
        img = np.transpose(img, (1, 2, 0))

    return img

def load_dncnn(model_path, device=None):

    # noise_level_img = 25             # noise level for noisy image
    # model_name = 'dncnn_25'          # 'dncnn_15' | 'dncnn_25' | 'dncnn_50' | 'dncnn_gray_blind' | 'dncnn_color_blind' | 'dncnn3'
    # testset_name = 'bsd68'           # test set, 'bsd68' | 'set12'
    # need_degradation = True          # default: True
    # x8 = False                       # default: False, x8 to boost performance
    # show_img = False                 # default: False
    #
    # task_current = 'dn'       # 'dn' for denoising | 'sr' for super-resolution
    # sf = 1                    # unused for denoising
    # if 'color' in model_name:
    #     n_channels = 3        # fixed, 1 for grayscale image, 3 for color image
    # else:
    n_channels = 1        # fixed for grayscale image
    # if model_name in ['dncnn_gray_blind', 'dncnn_color_blind', 'dncnn3']:
    #     nb = 20               # fixed
    # else:
    nb = 17               # fixed
    # model_pool = 'model_zoo'  # fixed
    # testsets = 'testsets'     # fixed
    # results = 'results'       # fixed
    # result_name = testset_name + '_' + model_name     # fixed
    # border = sf if task_current == 'sr' else 0        # shave boader to calculate PSNR and SSIM
    # model_path = os.path.join(model_pool, model_name+'.pth')


    # if H_path == L_path:
    #     need_degradation = True
    # logger_name = result_name
    # utils_logger.logger_info(logger_name, log_path=os.path.join(E_path, logger_name+'.log'))
    # logger = logging.getLogger(logger_name)

    # need_H = True if H_path is not None else False
    if device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ----------------------------------------
    # load model
    # ----------------------------------------

    model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='R')
    # model = net(in_nc=n_channels, out_nc=n_channels, nc=64, nb=nb, act_mode='BR')  # use this if BN is not merged by utils_bnorm.merge_bn(model)
    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    return model
    # logger.info('Model path: {:s}'.format(model_path))
    # number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    # logger.info('Params number: {}'.format(number_parameters))
#
#     test_results = OrderedDict()
#     test_results['psnr'] = []
#     test_results['ssim'] = []
#
#     logger.info('model_name:{}, image sigma:{}'.format(model_name, noise_level_img))
#     logger.info(L_path)
#     L_paths = util.get_image_paths(L_path)
#     H_paths = util.get_image_paths(H_path) if need_H else None
#
#     for idx, img in enumerate(L_paths):
#
#         # ------------------------------------
#         # (1) img_L
#         # ------------------------------------
#
#         img_name, ext = os.path.splitext(os.path.basename(img))
#         # logger.info('{:->4d}--> {:>10s}'.format(idx+1, img_name+ext))
#         img_L = util.imread_uint(img, n_channels=n_channels)
#         img_L = util.uint2single(img_L)
#
#         if need_degradation:  # degradation process
#             np.random.seed(seed=0)  # for reproducibility
#             img_L += np.random.normal(0, noise_level_img/255., img_L.shape)
#
#         util.imshow(util.single2uint(img_L), title='Noisy image with noise level {}'.format(noise_level_img)) if show_img else None
#
#         img_L = util.single2tensor4(img_L)
#         img_L = img_L.to(device)
#
#         # ------------------------------------
#         # (2) img_E
#         # ------------------------------------
#
#         if not x8:
#             img_E = model(img_L)
#         else:
#             img_E = utils_model.test_mode(model, img_L, mode=3)
#
#         img_E = util.tensor2uint(img_E)
#
#         if need_H:
#
#             # --------------------------------
#             # (3) img_H
#             # --------------------------------
#
#             img_H = util.imread_uint(H_paths[idx], n_channels=n_channels)
#             img_H = img_H.squeeze()
#
#             # --------------------------------
#             # PSNR and SSIM
#             # --------------------------------
#
#             psnr = util.calculate_psnr(img_E, img_H, border=border)
#             ssim = util.calculate_ssim(img_E, img_H, border=border)
#             test_results['psnr'].append(psnr)
#             test_results['ssim'].append(ssim)
#             logger.info('{:s} - PSNR: {:.2f} dB; SSIM: {:.4f}.'.format(img_name+ext, psnr, ssim))
#             util.imshow(np.concatenate([img_E, img_H], axis=1), title='Recovered / Ground-truth') if show_img else None
#
#         # ------------------------------------
#         # save results
#         # ------------------------------------
#
#         util.imsave(img_E, os.path.join(E_path, img_name+ext))
#
#     if need_H:
#         ave_psnr = sum(test_results['psnr']) / len(test_results['psnr'])
#         ave_ssim = sum(test_results['ssim']) / len(test_results['ssim'])
#         logger.info('Average PSNR/SSIM(RGB) - {} - PSNR: {:.2f} dB; SSIM: {:.4f}'.format(result_name, ave_psnr, ave_ssim))
#
# if __name__ == '__main__':
#
#     main()
