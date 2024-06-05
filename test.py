import torch
import numpy as np
from collections import OrderedDict
from Network import Net
import utils
from tqdm import tqdm
import glob
import time
import os
from train import receive_arg


def get_h_w_f(filename):
    WxH = filename.split('_')[-2]   
    frame_nums = int((filename.split('_')[-1]).split('.')[0])    
    width = int(WxH.split('x')[0])       
    height = int(WxH.split('x')[1])        
    return height, width, frame_nums



def test_one_video(model, opts_dict, lq_yuv_file, raw_yuv_file, scale=1):
    h, w, nfs = get_h_w_f(raw_yuv_file)
    raw_y = utils.import_yuv(seq_path=raw_yuv_file, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True)  
    lq_y = utils.import_yuv(seq_path=lq_yuv_file, h=h, w=w, tot_frm=nfs, start_frm=0, only_y=True)
    raw_y = raw_y.astype(np.float32) / 255.
    lq_y = lq_y.astype(np.float32) / 255.


    pbar = tqdm(total=nfs, ncols=80)
    criterion_psnr = utils.PSNR()
    unit = 'dB'
    ori_psnr_counter = utils.Counter()
    enh_psnr_counter = utils.Counter()

    criterion_ssim = utils.SSIM()
    ori_ssim_counter = utils.Counter()
    enh_ssim_counter = utils.Counter()
    with torch.no_grad():  
        for idx in range(nfs): 
            idx_list = list(range(idx - opts_dict['network']['radius'], idx + opts_dict['network']['radius'] + 1))  
            idx_list = np.clip(idx_list, 0, nfs-1)  
            input_data = []
            for idx_ in idx_list:
                input_data.append(lq_y[idx_])
            input_data = torch.from_numpy(np.array(input_data))
            input_data = torch.unsqueeze(input_data, 0).cuda()  


            b, tc, h, w = input_data.shape  # torch.Size([1, opts_dict['network']['radius']*2+1, h, w]) 
            output_data = input_data.new_zeros((1, h*scale, w*scale))

            max_h, max_w = h // 1, w // 1  
            for c_h in range(h // max_h):  
                for c_w in range(w // max_w):
                    result_data = model(input_data[:, :, c_h*max_h: (c_h+1)*max_h, c_w*max_w: (c_w+1)*max_w].contiguous())  
                    torch.cuda.empty_cache()  
                    output_data[:, c_h*max_h*scale: (c_h+1)*max_h*scale, c_w*max_w*scale: (c_w+1)*max_w*scale] = result_data[0, :, :, :]

            gt_frm = torch.from_numpy(raw_y[idx:idx+1]).cuda()
            batch_pre_psnr = criterion_psnr(input_data[0, opts_dict['network']['radius']:opts_dict['network']['radius']+1, ...], gt_frm)
            batch_aft_psnr = criterion_psnr(output_data, gt_frm)
            ori_psnr_counter.accum(volume=batch_pre_psnr)
            enh_psnr_counter.accum(volume=batch_aft_psnr)


            batch_pre_ssim = criterion_ssim(input_data[0, opts_dict['network']['radius']:opts_dict['network']['radius']+1, ...], gt_frm)
            batch_aft_ssim = criterion_ssim(output_data, gt_frm)
            ori_ssim_counter.accum(volume=batch_pre_ssim)
            enh_ssim_counter.accum(volume=batch_aft_ssim)


            pbar.set_description("[{:.3f}]{:s} -> [{:.3f}]{:s}".format(batch_pre_psnr, unit, batch_aft_psnr, unit))
            pbar.update()
    pbar.close()
    print('cost of time:{:.2f}s'.format(time.time() - start_time))

    ori_psnr = ori_psnr_counter.get_ave()
    enh_psnr = enh_psnr_counter.get_ave()
    print('PSNR:  ave ori[{:.3f}]{:s}, enh[{:.3f}]{:s}, delta[{:.3f}]{:s}'.format(ori_psnr, unit, enh_psnr, unit, (enh_psnr - ori_psnr), unit))
    
    ori_ssim = ori_ssim_counter.get_ave()
    enh_ssim = enh_ssim_counter.get_ave()
    print('SSIM:  ave ori[%d], enh[%d], delta[%d]' % (ori_ssim*10000, enh_ssim*10000, (enh_ssim - ori_ssim)*10000))




if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    test_QP = 'QP37'
    ckp_path = sorted(glob.glob(f'./exp/{test_QP}' + '/*.pt'))[0]
    print(ckp_path)
    qp_yuv = sorted(glob.glob(f'/home/luodengyan/mfqe_datasets/YUV/test_qp_yuv/{test_QP}' + '/*.yuv'))
    raw_yuv = sorted(glob.glob('/home/luodengyan/mfqe_datasets/YUV/test_raw_yuv' + '/*.yuv'))


    opts_dict = receive_arg()
    model = Net(opts_dict=opts_dict['network'])
    start_time = time.time()
    checkpoint = torch.load(ckp_path)

    if 'module.' in list(checkpoint['state_dict'].keys())[0]:  # multi-gpu training
        new_state_dict = OrderedDict()
        for k, v in checkpoint['state_dict'].items():
            name = k[7:]  # remove module
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:  # single-gpu training
        model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()


    for test_file_num in range(len(raw_yuv)):  # 14, 15   17, 18 len(raw_yuv)  15, 17
        lq_yuv_file =  qp_yuv[test_file_num]
        raw_yuv_file = raw_yuv[test_file_num]
        print('*' * 70)
        print(lq_yuv_file)
        print(raw_yuv_file)
        print('*' * 70)

        test_one_video(model, opts_dict, lq_yuv_file, raw_yuv_file)
        print('> done.')
        print()