import argparse

from torch.utils.data import DataLoader

from data.select_datasets import select_data
from models import *
from utils import *


def load_model_weights(model, checkpoint_path, device):
    # 读取 checkpoint 文件，参数 checkpoint_path 表示模型权重路径。
    checkpoint = torch.load(checkpoint_path, map_location=device)
    # 兼容 {'model': state_dict} 这类保存格式，参数 checkpoint 表示已加载的权重对象。
    state_dict = checkpoint['model'] if isinstance(checkpoint, dict) and 'model' in checkpoint else checkpoint
    # 优先走新模型的部分加载逻辑，参数 model 表示待加载的网络。
    if hasattr(model, 'load_pretrained'):
        # 调用模型自定义的权重映射逻辑，三个返回值分别表示命中数、模型键数、checkpoint 键数。
        matched_count, model_key_count, checkpoint_key_count = model.load_pretrained(state_dict)
        # 打印部分加载统计，参数 matched_count、model_key_count、checkpoint_key_count 分别表示加载统计量。
        print('Loaded pretrained params: {:d}/{:d} model keys matched from {:d} checkpoint keys.'.format(
            matched_count, model_key_count, checkpoint_key_count))
        # 直接返回，避免再走严格加载。
        return
    # 对无自定义加载逻辑的模型保持原有严格加载行为，参数 state_dict 表示 checkpoint 权重字典。
    model.load_state_dict(state_dict, strict=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--opt', type=str, default='options/test/x4/test_braint_spsr_release_s4_d32_w5_n1.json')
    # parser.add_argument('--opt', type=str, default='options/test/x4/test_ixit_spsr_release_s4_d32_w5_n1.json')
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    args = parser.parse_args()
    with open(args.opt, 'r', encoding='utf-8') as f:
        json_str = f.read()

    opt = json.loads(json_str, object_pairs_hook=OrderedDict)
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device = args.device

    # set up model
    if os.path.exists(opt['path']['pretrained_netG']):
        print('loading model from {}'.format(opt['path']['pretrained_netG']))
    logger = get_logger('test', os.path.join(os.path.dirname(opt['path']['pretrained_netG']), '..', 'test.log'))
    logger.info(opt)

    model = select_G(opt)

    # 加载预训练权重，参数 model 表示网络实例，参数 pretrained_netG 表示权重路径。
    load_model_weights(model, opt['path']['pretrained_netG'], device)

    model = model.to(device)
    opt['datasets']['test']['scale'] = opt['scale']
    opt['datasets']['test']['n_channels'] = opt['n_channels']
    opt['datasets']['test']['img_size'] = opt['netG']['img_size']
    opt['datasets']['test']['maxn'] = opt['maxn']
    opt['datasets']['test']['minn'] = opt['minn']
    opt['datasets']['test']['phase'] = 'test'

    seed = 3407

    print('Random seed: {}'.format(seed))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    test_set = select_data(opt['datasets']['test'])
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1, drop_last=False, pin_memory=True)

    avg_psnr = []
    avg_ssim = []
    n_channels = 1
    minn = opt['minn']
    model.eval()
    for idx, data in enumerate(test_loader):
        # read image
        image_name_ext = os.path.basename(data['path'][0]).replace('PD', 'T2').replace('T1', 'T2')
        result = []
        maxn = min(data['L'][0].shape[1], opt['maxn'])
        for j in range(minn, maxn, n_channels):
            with torch.no_grad():
                lr = data['L'][0][:, j:min(j + n_channels, maxn), :, :].float().to(device)
                ref = data['L'][1][:, j:min(j + n_channels, maxn), :, :].float().to(device)
                E_img = model([lr, ref])[0]
                # E_img = F.interpolate(lr, size=(240, 240))
                E_img = tensor2single(E_img)
                # E_img = E_img.squeeze().float().cpu().numpy()
                result.append(E_img)
        result = np.array(result)
        current_psnr = psnr(result, data['H'][0].squeeze().cpu().numpy()[minn:maxn])
        current_ssim = ssim(result, data['H'][0].squeeze().cpu().numpy()[minn:maxn])
        logger.info('{:->4d}--> {:>10s} | {:<4.2f}dB {:<4.4f}'.format(idx, image_name_ext, current_psnr, current_ssim))
        avg_psnr.append(current_psnr)
        avg_ssim.append(current_ssim)

    logger.info('-- Average PSNR/SSIM: {:.4f} dB|{:.4f}'.format(np.mean(avg_psnr), np.mean(avg_ssim)))


if __name__ == '__main__':
    main()
