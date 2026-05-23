import nibabel as nib
from torch.utils.data import Dataset

from utils import *


class DatasetBrainSRT(Dataset):
    '''
    Get L/H for SISR.
    must paths_H and path_L both are provided.
    '''

    def __init__(self, opt):
        super(DatasetBrainSRT, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.data_root = opt['dataroot_H']
        self.scale = opt['scale'] if opt['scale'] else 4
        self.paths = get_image_paths(self.data_root)

    def __getitem__(self, index):

        volumepath = self.paths[index]
        volume = nib.load(volumepath)
        volumeIn = np.array([volume.get_fdata()])
        volumeIn_t1 = volumeIn[:, :, :, :, 0].squeeze()
        volumeIn_t2 = volumeIn[:, :, :, :, 2].squeeze()

        max_d, min_d = max(volumeIn_t1.reshape(-1).max(), volumeIn_t2.reshape(-1).max()), min(
            volumeIn_t1.reshape(-1).min(), volumeIn_t2.reshape(-1).min())
        if (max_d - min_d) != 0:
            volumeIn_t1 = (volumeIn_t1 - volumeIn_t1.reshape(-1).min()) / (max_d - min_d)
            volumeIn_t2 = (volumeIn_t2 - volumeIn_t2.reshape(-1).min()) / (max_d - min_d)

        H, W, C = volumeIn_t2.shape
        volumeDown_t2 = self.degrade(volumeIn_t2).transpose(2, 1, 0)
        volumeDown_t1 = self.degrade(volumeIn_t1).transpose(2, 1, 0)

        idx = random.randint(0, C - 1)
        if self.opt['phase'] == 'train':
            volumeIn_t1 = volumeIn_t1[:, :, idx]
            volumeIn_t2 = volumeIn_t2[:, :, idx]
            volumeDown_t2 = volumeDown_t2[:, :, idx]
            volumeDown_t1 = volumeDown_t1[:, :, idx]

        volumeDown_t2 = single2tensor3(volumeDown_t2)
        volumeDown_t1 = single2tensor3(volumeDown_t1)
        volumeIn_t2 = single2tensor3(volumeIn_t2)
        volumeIn_t1 = single2tensor3(volumeIn_t1)
        # L:[I_in,Rc] H:[HR,HR]
        return {'L': [volumeDown_t2, volumeIn_t1],
                'H': [volumeIn_t2, get_gradient(volumeIn_t2.unsqueeze(1)).squeeze(1)], 'path': self.paths[index]}

    def __len__(self):
        return len(self.paths)

    def degrade(self, hr_data):
        norm_d = max(hr_data.reshape(-1)) - min(hr_data.reshape(-1))
        if self.scale == 4:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 90: 150, 90: 150]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)

        if self.scale == 3:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 80: 160, 80: 160]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)

        if self.scale == 2:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 60: 180, 60: 180]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
        if img_out.max() - img_out.min() != 0:
            img_out = (img_out - min(img_out.reshape(-1))) / (max(img_out.reshape(-1)) - min(img_out.reshape(-1)))
        return img_out


class DatasetIXIT(Dataset):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.data_root = opt['dataroot_H']
        self.HR_paths = get_image_paths(self.data_root)
        self.scale = opt['scale'] if opt['scale'] else 4
        self.img_size = opt['img_size']
        # 记录归一化模式，参数表示是否使用联合最值、独立最值或独立分位数归一化。
        self.normalize_mode = opt.get('normalize_mode', 'joint_minmax')
        # 记录稳健分位数归一化的上下分位点，两个参数分别表示低分位和高分位。
        self.normalize_percentiles = opt.get('normalize_percentiles', [0.5, 99.5])
        # 记录退化后的 LR 是否再次独立归一化，参数为布尔值。
        self.lr_renormalize = opt.get('lr_renormalize', True)

    def _normalize_minmax(self, volume):
        # 读取当前体数据的最小值，参数 volume 表示待归一化的三维体数据。
        min_value = float(volume.min())
        # 读取当前体数据的最大值，参数 volume 表示待归一化的三维体数据。
        max_value = float(volume.max())
        # 计算当前体数据的动态范围，参数 max_value 和 min_value 分别表示上下界。
        scale_value = max_value - min_value
        # 判断动态范围是否为 0，避免除零。
        if scale_value == 0:
            # 返回 float32 结果，参数 copy=False 表示尽量避免额外拷贝。
            return volume.astype(np.float32, copy=False)
        # 按最值把数据缩放到 0 到 1，参数 scale_value 表示缩放分母。
        normalized = (volume - min_value) / scale_value
        # 返回 float32 结果，参数 copy=False 表示尽量复用内存。
        return normalized.astype(np.float32, copy=False)

    def _normalize_percentile(self, volume):
        # 读取低分位点，参数 normalize_percentiles[0] 表示低端截断比例。
        low_percentile = float(self.normalize_percentiles[0])
        # 读取高分位点，参数 normalize_percentiles[1] 表示高端截断比例。
        high_percentile = float(self.normalize_percentiles[1])
        # 计算低分位阈值，参数 volume 表示待归一化体数据。
        low_value = float(np.percentile(volume, low_percentile))
        # 计算高分位阈值，参数 volume 表示待归一化体数据。
        high_value = float(np.percentile(volume, high_percentile))
        # 判断分位区间是否退化，避免除零。
        if high_value <= low_value:
            # 回退到普通最值归一化，参数 volume 表示待处理体数据。
            return self._normalize_minmax(volume)
        # 先把异常高低值裁剪到稳健区间，参数 low_value 和 high_value 表示裁剪边界。
        clipped = np.clip(volume, low_value, high_value)
        # 按稳健区间缩放到 0 到 1，参数 high_value 和 low_value 分别表示上下界。
        normalized = (clipped - low_value) / (high_value - low_value)
        # 返回 float32 结果，参数 copy=False 表示尽量复用内存。
        return normalized.astype(np.float32, copy=False)

    def _normalize_pair(self, volume_t1, volume_t2):
        # 读取归一化模式，参数 normalize_mode 表示当前采用的策略。
        normalize_mode = self.normalize_mode
        # 保留旧版联合最值归一化，参数 volume_t1 和 volume_t2 分别表示 T1 和 T2 体数据。
        if normalize_mode == 'joint_minmax':
            # 读取联合最大值，参数 volume_t1 和 volume_t2 分别提供两个模态。
            max_value = max(float(volume_t1.max()), float(volume_t2.max()))
            # 读取联合最小值，参数 volume_t1 和 volume_t2 分别提供两个模态。
            min_value = min(float(volume_t1.min()), float(volume_t2.min()))
            # 计算联合动态范围，参数 max_value 和 min_value 分别表示上下界。
            scale_value = max_value - min_value
            # 判断动态范围是否为 0，避免除零。
            if scale_value == 0:
                # 直接返回 float32 的 T1，参数 copy=False 表示尽量复用内存。
                volume_t1 = volume_t1.astype(np.float32, copy=False)
                # 直接返回 float32 的 T2，参数 copy=False 表示尽量复用内存。
                volume_t2 = volume_t2.astype(np.float32, copy=False)
            else:
                # 用联合区间缩放 T1，参数 scale_value 表示缩放分母。
                volume_t1 = ((volume_t1 - min_value) / scale_value).astype(np.float32, copy=False)
                # 用联合区间缩放 T2，参数 scale_value 表示缩放分母。
                volume_t2 = ((volume_t2 - min_value) / scale_value).astype(np.float32, copy=False)
            # 返回联合归一化后的两个模态。
            return volume_t1, volume_t2
        # 对每个模态分别做最值归一化，参数 volume_t1 和 volume_t2 分别表示两个模态。
        if normalize_mode == 'independent_minmax':
            # 返回独立最值归一化后的 T1 和 T2。
            return self._normalize_minmax(volume_t1), self._normalize_minmax(volume_t2)
        # 对每个模态分别做稳健分位数归一化，参数 volume_t1 和 volume_t2 分别表示两个模态。
        if normalize_mode == 'independent_percentile':
            # 返回独立分位数归一化后的 T1 和 T2。
            return self._normalize_percentile(volume_t1), self._normalize_percentile(volume_t2)
        # 对未知模式抛出异常，参数 normalize_mode 表示传入的非法配置。
        raise ValueError('Unknown normalize_mode: {}'.format(normalize_mode))

    def __getitem__(self, index):
        volumepath_t1 = self.HR_paths[index]
        volumepath_t2 = volumepath_t1.replace('t1', 't2').replace('T1', 'T2')
        volume_t1 = nib.load(volumepath_t1)
        volume_t2 = nib.load(volumepath_t2)

        volumeIn_t1 = np.array([volume_t1.get_fdata()]).squeeze()
        volumeIn_t2 = np.array([volume_t2.get_fdata()]).squeeze()
        # 256,256,130 -> 240,240,130
        volumeIn_t1 = cv2.resize(volumeIn_t1, (self.img_size * self.scale, self.img_size * self.scale))
        # 256,256,130*2 -> 240,240,260
        volumeIn_t2 = volumeIn_t2.squeeze()
        volumeIn_t2 = cv2.resize(volumeIn_t2, (self.img_size * self.scale, self.img_size * self.scale))

        H, W, C = volumeIn_t2.shape

        idx = random.randint(0, C - 1)
        # 按配置对 T1 和 T2 做归一化，参数 volumeIn_t1 和 volumeIn_t2 分别表示两个模态体数据。
        volumeIn_t1, volumeIn_t2 = self._normalize_pair(volumeIn_t1, volumeIn_t2)
        # 仅退化目标模态 T2，参数 lr_renormalize 表示是否对退化后 LR 再做独立归一化。
        volumeDown_t2 = self.degrade(volumeIn_t2, renormalize=self.lr_renormalize).transpose(2, 1, 0)

        if self.opt['phase'] == 'train':
            volumeIn_t1 = volumeIn_t1[:, :, idx]
            volumeIn_t2 = volumeIn_t2[:, :, idx]
            volumeDown_t2 = volumeDown_t2[:, :, idx]

        volumeDown_t2 = single2tensor3(volumeDown_t2)
        volumeIn_t2 = single2tensor3(volumeIn_t2)
        volumeIn_t1 = single2tensor3(volumeIn_t1)

        # L:[I_in,ref] H:[HR,HR]
        sample = {'L': [volumeDown_t2, volumeIn_t1],
                  'H': [volumeIn_t2, get_gradient(volumeIn_t2.unsqueeze(1)).squeeze(1)],
                  'path': self.HR_paths[index]}
        return sample

    def __len__(self):
        return len(self.HR_paths)

    def degrade(self, hr_data, renormalize=True):
        norm_d = max(hr_data.reshape(-1)) - min(hr_data.reshape(-1))
        if self.scale == 4:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 90: 150, 90: 150]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)

        if self.scale == 3:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 80: 160, 80: 160]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)

        if self.scale == 2:
            imgfft = np.fft.fft2(hr_data.transpose(2, 1, 0))
            imgfft = np.fft.fftshift(imgfft)
            imgfft = imgfft[:, 60: 180, 60: 180]
            imgfft = np.fft.ifftshift(imgfft)
            imgifft = np.fft.ifft2(imgfft)
            img_out = abs(imgifft)
        if renormalize and img_out.max() - img_out.min() != 0:
            img_out = (img_out - min(img_out.reshape(-1))) / (max(img_out.reshape(-1)) - min(img_out.reshape(-1)))
        return img_out


class DatasetIXISROriWM(Dataset):
    '''
    Get L/H for SISR.
    must paths_H and path_L both are provided.
    '''

    def __init__(self, opt):
        super(DatasetIXISROriWM, self).__init__()
        self.opt = opt
        self.data_root = opt['dataroot_H']
        self.scale = opt['scale'] if opt['scale'] else 4
        self.HR_paths = get_image_paths(opt['dataroot_H'])
        self.LR_paths = get_image_paths(opt['dataroot_L'])

    def __getitem__(self, index):
        # 或者pd做t1
        volumeDownpath_t2 = self.LR_paths[index].replace('t1', 't2').replace('T1', 'T2').replace('PD', 'T2')
        volumepath_t2 = self.HR_paths[index].replace('t1', 't2').replace('T1', 'T2').replace('PD', 'T2')
        volumeDown_t1 = np.load(self.LR_paths[index])
        volumeDown_t2 = np.load(volumeDownpath_t2)
        volumeIn_t2 = np.load(volumepath_t2)
        volumeIn_t1 = np.load(self.HR_paths[index])
        H, W, C = volumeDown_t2.shape

        idx = random.randint(0, C - 1)
        if self.opt['phase'] == 'train':
            volumeIn_t2 = volumeIn_t2[:, :, idx]
            volumeIn_t1 = volumeIn_t1[:, :, idx]
            volumeDown_t2 = volumeDown_t2[:, :, idx]
            volumeDown_t1 = volumeDown_t1[:, :, idx]

        volumeDown_t2 = single2tensor3(volumeDown_t2)
        volumeIn_t2 = single2tensor3(volumeIn_t2)
        volumeIn_t1 = single2tensor3(volumeIn_t1)
        volumeDown_t1 = single2tensor3(volumeDown_t1)
        # L:[I_in,Rc] H:[HR,HR]
        return {'L': [volumeDown_t2, volumeDown_t1],
                'H': [volumeIn_t2, get_gradient(volumeIn_t2.unsqueeze(1)).squeeze(1)],
                'path': self.HR_paths[index]}

    def __len__(self):
        return len(self.HR_paths)
