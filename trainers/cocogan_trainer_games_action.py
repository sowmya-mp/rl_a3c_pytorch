"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from .cocogan_nets import *
from .init import *
from .helpers import get_model_list, _compute_fake_acc, _compute_true_acc
from .cocogan_trainer_games import *
import torch
import torch.nn as nn
import os
import itertools
import pdb


class COCOGANGAMESACTIONTrainer(COCOGANGAMESTrainer):
  def __init__(self, hyperparameters):
    super(COCOGANGAMESACTIONTrainer, self).__init__(hyperparameters)

  def gen_update(self, images_a, images_b, labels_a, labels_b, hyperparameters):
    self.gen.zero_grad()
    if self.use_xy:
        images_a_xy_0 = torch.cat((images_a[0], self.xy), 1)
        images_b_xy_0 = torch.cat((images_b[0], self.xy), 1)
        images_a_xy_1 = torch.cat((images_a[1], self.xy), 1)
        images_b_xy_1 = torch.cat((images_b[1], self.xy), 1)
    else:
        images_a_xy_0 = images_a[0]
        images_b_xy_0 = images_b[0]
        images_a_xy_1 = images_a[1]
        images_b_xy_1 = images_b[1]
    x_aa_0, x_ba_0, x_ab_0, x_bb_0, shared_0 = self.gen(images_a_xy_0, images_b_xy_0)
    x_aa_1, x_ba_1, x_ab_1, x_bb_1, shared_1 = self.gen(images_a_xy_1, images_b_xy_1)

    if self.use_xy:
        x_ba_xy_0 = torch.cat((x_ba_0, self.xy), 1)
        x_ab_xy_0 = torch.cat((x_ab_0, self.xy), 1)
        x_ba_xy_1 = torch.cat((x_ba_1, self.xy), 1)
        x_ab_xy_1 = torch.cat((x_ab_1, self.xy), 1)
    else:
        x_ba_xy_0 = x_ba_0
        x_ab_xy_0 = x_ab_0
        x_ba_xy_1 = x_ba_1
        x_ab_xy_1 = x_ab_1
    x_bab_0, shared_bab_0 = self.gen.forward_a2b(x_ba_xy_0)
    x_aba_0, shared_aba_0 = self.gen.forward_b2a(x_ab_xy_0)
    x_bab_1, shared_bab_1 = self.gen.forward_a2b(x_ba_xy_1)
    x_aba_1, shared_aba_1 = self.gen.forward_b2a(x_ab_xy_1)

    x_ba = torch.cat((x_ba_0, x_ba_1), 1)
    x_ab = torch.cat((x_ab_0, x_ab_1), 1)

    outs_a, outs_b = self.dis(x_ba,x_ab)
    for it, (out_a, out_b) in enumerate(itertools.izip(outs_a, outs_b)):
      all_labels_a = labels_a.repeat(out_a.size(0)//labels_a.size(0)).resize(out_a.size(0)//labels_a.size(0), labels_a.size(0)).transpose(1,0).resize(out_a.size(0))
      all_labels_b = labels_b.repeat(out_b.size(0)//labels_b.size(0)).resize(out_b.size(0)//labels_b.size(0), labels_b.size(0)).transpose(1,0).resize(out_b.size(0))
      if it==0:
        ad_loss_a = nn.functional.cross_entropy(out_a, all_labels_b)
        ad_loss_b = nn.functional.cross_entropy(out_b, all_labels_a)
      else:
        ad_loss_a += nn.functional.cross_entropy(out_a, all_labels_b)
        ad_loss_b += nn.functional.cross_entropy(out_b, all_labels_a)

    enc_loss  = self._compute_kl(shared_0)
    enc_bab_loss = self._compute_kl(shared_bab_0)
    enc_aba_loss = self._compute_kl(shared_aba_0)
    ll_loss_a = self.ll_loss_criterion_a(x_aa_0, images_a[0])
    ll_loss_b = self.ll_loss_criterion_b(x_bb_0, images_b[0])
    ll_loss_aba = self.ll_loss_criterion_a(x_aba_0, images_a[0])
    ll_loss_bab = self.ll_loss_criterion_b(x_bab_0, images_b[0])
    total_loss = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b) + \
                 hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b) + \
                 hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab) + \
                 hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss) + \
                 hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss)
    total_loss.backward()
    self.gen_opt.step()
    self.gen_enc_loss = enc_loss.data.cpu().numpy()[0]
    self.gen_enc_bab_loss = enc_bab_loss.data.cpu().numpy()[0]
    self.gen_enc_aba_loss = enc_aba_loss.data.cpu().numpy()[0]
    self.gen_ad_loss_a = ad_loss_a.data.cpu().numpy()[0]
    self.gen_ad_loss_b = ad_loss_b.data.cpu().numpy()[0]
    self.gen_ll_loss_a = ll_loss_a.data.cpu().numpy()[0]
    self.gen_ll_loss_b = ll_loss_b.data.cpu().numpy()[0]
    self.gen_ll_loss_aba = ll_loss_aba.data.cpu().numpy()[0]
    self.gen_ll_loss_bab = ll_loss_bab.data.cpu().numpy()[0]
    self.gen_total_loss = total_loss.data.cpu().numpy()[0]
    return (x_aa_0, x_ba_0, x_ab_0, x_bb_0, x_aba_0, x_bab_0)

  def dis_update(self, images_a, images_b, labels_a, labels_b, hyperparameters):
    self.dis.zero_grad()
    if self.use_xy:
        images_a_xy_0 = torch.cat((images_a[0], self.xy), 1)
        images_b_xy_0 = torch.cat((images_b[0], self.xy), 1)
        images_a_xy_1 = torch.cat((images_a[1], self.xy), 1)
        images_b_xy_1 = torch.cat((images_b[1], self.xy), 1)
        images_a_xy = torch.cat(images_a, 1)
        images_b_xy = torch.cat(images_b, 1)
    else:
        images_a_xy_0 = images_a[0]
        images_b_xy_0 = images_b[0]
        images_a_xy_1 = images_a[1]
        images_b_xy_1 = images_b[1]
    x_aa_0, x_ba_0, x_ab_0, x_bb_0, shared_0 = self.gen(images_a_xy_0, images_b_xy_0)
    x_aa_1, x_ba_1, x_ab_1, x_bb_1, shared_1 = self.gen(images_a_xy_1, images_b_xy_1)


    x_ba = torch.cat((x_ba_0, x_ba_1), 1)
    x_ab = torch.cat((x_ab_0, x_ab_1), 1)

    data_a = torch.cat((images_a_xy, x_ba), 0)
    data_b = torch.cat((images_b_xy, x_ab), 0)
    res_a, res_b = self.dis(data_a,data_b)
    # res_true_a, res_true_b = self.dis(images_a,images_b)
    # res_fake_a, res_fake_b = self.dis(x_ba, x_ab)
    for it, (out_a, out_b) in enumerate(itertools.izip(res_a, res_b)):
      out_true_a, out_fake_a = torch.split(out_a, out_a.size(0) // 2, 0)
      out_true_b, out_fake_b = torch.split(out_b, out_b.size(0) // 2, 0)
      out_true_n = out_true_a.size(0)
      out_fake_n = out_fake_a.size(0)
      all0 = Variable(torch.LongTensor(out_fake_n).fill_(0).cuda(self.gpu))

      all_labels_a = labels_a.repeat(out_true_a.size(0)//labels_a.size(0)).resize(out_true_a.size(0)//labels_a.size(0), labels_a.size(0)).transpose(1,0).resize(out_true_a.size(0))
      all_labels_b = labels_b.repeat(out_true_b.size(0)//labels_b.size(0)).resize(out_true_b.size(0)//labels_b.size(0), labels_b.size(0)).transpose(1,0).resize(out_true_b.size(0))

      ad_true_loss_a = nn.functional.cross_entropy(out_true_a, all_labels_a)
      ad_true_loss_b = nn.functional.cross_entropy(out_true_b, all_labels_b)
      ad_fake_loss_a = nn.functional.cross_entropy(out_fake_a, all0)
      ad_fake_loss_b = nn.functional.cross_entropy(out_fake_b, all0)
      if it==0:
        ad_loss_a = ad_true_loss_a + ad_fake_loss_a
        ad_loss_b = ad_true_loss_b + ad_fake_loss_b
      else:
        ad_loss_a += ad_true_loss_a + ad_fake_loss_a
        ad_loss_b += ad_true_loss_b + ad_fake_loss_b
      true_a_acc = _compute_true_acc(out_true_a)
      true_b_acc = _compute_true_acc(out_true_b)
      fake_a_acc = _compute_fake_acc(out_fake_a)
      fake_b_acc = _compute_fake_acc(out_fake_b)
      exec( 'self.dis_true_acc_%d = 0.5 * (true_a_acc + true_b_acc)' %it)
      exec( 'self.dis_fake_acc_%d = 0.5 * (fake_a_acc + fake_b_acc)' %it)
    loss = hyperparameters['gan_w'] * ( ad_loss_a + ad_loss_b )
    loss.backward()
    self.dis_opt.step()
    self.dis_loss = loss.data.cpu().numpy()[0]
    return

  def assemble_outputs(self, images_a, images_b, network_outputs):
    images_a = self.normalize_image(images_a)
    images_b = self.normalize_image(images_b)
    x_aa = self.normalize_image(network_outputs[0])
    x_ba = self.normalize_image(network_outputs[1])
    x_ab = self.normalize_image(network_outputs[2])
    x_bb = self.normalize_image(network_outputs[3])
    x_aba = self.normalize_image(network_outputs[4])
    x_bab = self.normalize_image(network_outputs[5])
    return torch.cat((images_a[0:1, ::], x_aa[0:1, ::], x_ab[0:1, ::], x_aba[0:1, ::],
                      images_b[0:1, ::], x_bb[0:1, ::], x_ba[0:1, ::], x_bab[0:1, ::]), 3)

  def resume(self, snapshot_prefix):
    dirname = os.path.dirname(snapshot_prefix)
    last_model_name = get_model_list(dirname,"gen")
    if last_model_name is None:
      return 0
    self.gen.load_state_dict(torch.load(last_model_name))
    iterations = int(last_model_name[-12:-4])
    last_model_name = get_model_list(dirname, "dis")
    self.dis.load_state_dict(torch.load(last_model_name))
    print('Resume from iteration %d' % iterations)
    return iterations

  def save(self, snapshot_prefix, iterations):
    gen_filename = '%s_gen_%08d.pkl' % (snapshot_prefix, iterations + 1)
    dis_filename = '%s_dis_%08d.pkl' % (snapshot_prefix, iterations + 1)
    torch.save(self.gen.state_dict(), gen_filename)
    torch.save(self.dis.state_dict(), dis_filename)

  def cuda(self, gpu):
    self.gpu = gpu
    self.dis.cuda(gpu)
    self.gen.cuda(gpu)
    self.xy = self.xy.cuda(gpu)

  def normalize_image(self, x):
    return x[:,0:3,:,:]
