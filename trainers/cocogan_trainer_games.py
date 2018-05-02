"""
Copyright (C) 2017 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-ND 4.0 license (https://creativecommons.org/licenses/by-nc-nd/4.0/legalcode).
"""
from cocogan_nets import *
from init import *
from helpers import get_model_list, _compute_fake_acc, _compute_true_acc
import torch
from torch import autograd
import torch.nn as nn
import os
import itertools
from visualize import make_dot
import pdb


class COCOGANGAMESTrainer(nn.Module):
  def __init__(self, hyperparameters):
    super(COCOGANGAMESTrainer, self).__init__()
    self.use_cuda = False
    lr = hyperparameters['lr']
    # Initiate the networks
    exec( 'self.dis = %s(hyperparameters[\'dis\'])' % hyperparameters['dis']['name'])
    exec( 'self.gen = %s(hyperparameters[\'gen\'])' % hyperparameters['gen']['name'] )
    # Setup the optimizers
    self.dis_opt = torch.optim.Adam(self.dis.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    self.gen_opt = torch.optim.Adam(self.gen.parameters(), lr=lr, betas=(0.5, 0.999), weight_decay=0.0001)
    # Network weight initialization
    self.dis.apply(gaussian_weights_init)
    self.gen.apply(gaussian_weights_init)
    # Setup the loss function for training
    self.ll_loss_criterion_a = torch.nn.L1Loss()
    self.ll_loss_criterion_b = torch.nn.L1Loss()
    xy = self._create_xy_image()
    self.xy = xy.unsqueeze(0).expand(hyperparameters['batch_size'], xy.size(0), xy.size(1), xy.size(2))
    self.use_xy = 1 == hyperparameters['gen']['use_xy']

  def _create_xy_image(self, width=256):
    coordinates = list(itertools.product(range(width), range(width)))
    arr = (np.reshape(np.asarray(coordinates), newshape=[width, width, 2]) - width/2 ) / float((width/2))
    new_map = np.transpose(np.float32(arr), [2, 0, 1])
    xy = Variable(torch.from_numpy(new_map), requires_grad=False)
    return xy

  def _compute_kl(self, mu):
    # def _compute_kl(self, mu, sd):
    # mu_2 = torch.pow(mu, 2)
    # sd_2 = torch.pow(sd, 2)
    # encoding_loss = (mu_2 + sd_2 - torch.log(sd_2)).sum() / mu_2.size(0)
    # return encoding_loss
    mu_2 = torch.pow(mu, 2)
    encoding_loss = torch.mean(mu_2)
    return encoding_loss

  def gen_update(self, images_a, images_b, hyperparameters):
    if self.use_xy:
        images_a_xy = torch.cat((images_a, self.xy), 1)
        images_b_xy = torch.cat((images_b, self.xy), 1)
    else:
        images_a_xy = images_a
        images_b_xy = images_b
    x_aa, x_ba, x_ab, x_bb, shared = self.gen(images_a_xy, images_b_xy)

    if self.use_xy:
        x_ba_xy = torch.cat((x_ba, self.xy), 1)
        x_ab_xy = torch.cat((x_ab, self.xy), 1)
    else:
        x_ba_xy = x_ba
        x_ab_xy = x_ab
    x_bab, shared_bab = self.gen.forward_a2b(x_ba_xy)
    x_aba, shared_aba = self.gen.forward_b2a(x_ab_xy)
    outs_a, outs_b = self.dis(x_ba,x_ab)
    for it, (out_a, out_b) in enumerate(itertools.izip(outs_a, outs_b)):
      #outputs_a = nn.functional.sigmoid(out_a)
      #outputs_b = nn.functional.sigmoid(out_b)
      #all_ones = Variable(torch.ones((outputs_a.size(0))).cuda(self.gpu))
      if it==0:
        ad_loss_a = -out_a.mean()
        ad_loss_b = -out_b.mean()
      else:
        ad_loss_a += -out_a.mean()
        ad_loss_b += -out_b.mean()

    enc_loss  = self._compute_kl(shared)
    enc_bab_loss = self._compute_kl(shared_bab)
    enc_aba_loss = self._compute_kl(shared_aba)
    ll_loss_a = self.ll_loss_criterion_a(x_aa, images_a)
    ll_loss_b = self.ll_loss_criterion_b(x_bb, images_b)
    ll_loss_aba = self.ll_loss_criterion_a(x_aba, images_a)
    ll_loss_bab = self.ll_loss_criterion_b(x_bab, images_b)

    total_loss = hyperparameters['gan_w'] * (ad_loss_a + ad_loss_b) + \
                 hyperparameters['ll_direct_link_w'] * (ll_loss_a + ll_loss_b) + \
                 hyperparameters['ll_cycle_link_w'] * (ll_loss_aba + ll_loss_bab) + \
                 hyperparameters['kl_direct_link_w'] * (enc_loss + enc_loss) + \
                 hyperparameters['kl_cycle_link_w'] * (enc_bab_loss + enc_aba_loss)
    self.gen.zero_grad()
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
    #pdb.set_trace()
    return (x_aa, x_ba, x_ab, x_bb, x_aba, x_bab)

  def dis_update(self, images_a, images_b, hyperparameters):
    if self.use_xy:
        images_a_xy = torch.cat((images_a, self.xy), 1)
        images_b_xy = torch.cat((images_b, self.xy), 1)
    else:
        images_a_xy = images_a
        images_b_xy = images_b
    x_aa, x_ba, x_ab, x_bb, shared = self.gen(images_a_xy, images_b_xy)

    data_a = torch.cat((images_a, x_ba), 0)
    data_b = torch.cat((images_b, x_ab), 0)
    #res_a, res_b = self.dis(data_a,data_b)
    #pdb.set_trace()
    res_true_a, res_true_b = self.dis(images_a, images_b)
    res_fake_a, res_fake_b = self.dis(x_ba, x_ab)

    # res_true_a, res_true_b = self.dis(images_a,images_b)
    # res_fake_a, res_fake_b = self.dis(x_ba, x_ab)
    if 0 != hyperparameters['gamma_js_regularization']:
      js_reg_a = self.js_regularization(res_true_a[0], images_a, res_fake_a[0], x_ba)
      js_reg_b = self.js_regularization(res_true_b[0], images_b, res_fake_b[0], x_ab)

    for it, (this_true_a, this_true_b, this_fake_a, this_fake_b, in_a, in_b, fake_a, fake_b) in enumerate(itertools.izip(res_true_a, res_true_b, res_fake_a, res_fake_b, images_a, images_b, x_ba, x_ab)):
      #pdb.set_trace()
      out_true_a, out_fake_a = nn.functional.sigmoid(this_true_a), nn.functional.sigmoid(this_fake_a)
      out_true_b, out_fake_b = nn.functional.sigmoid(this_true_b), nn.functional.sigmoid(this_fake_b)

      out_true_n = out_true_a.size(0)
      out_fake_n = out_fake_a.size(0)
      all1 = Variable(torch.ones((out_true_n)).cuda(self.gpu))
      all0 = Variable(torch.zeros((out_fake_n)).cuda(self.gpu))
      ad_true_loss_a = nn.functional.binary_cross_entropy(out_true_a, all1)
      ad_true_loss_b = nn.functional.binary_cross_entropy(out_true_b, all1)
      ad_fake_loss_a = nn.functional.binary_cross_entropy(out_fake_a, all0)
      ad_fake_loss_b = nn.functional.binary_cross_entropy(out_fake_b, all0)
      if it==0:
        ad_loss_a = (this_fake_a.mean() - this_true_a.mean())
        ad_loss_b = (this_fake_b.mean() - this_true_b.mean())
        panalty_a, panalty_b = self.calc_gradient_penalty(in_a, fake_a, in_b, fake_b)
        ad_loss_a += panalty_a
        ad_loss_b += panalty_b
      else:
        ad_loss_a += (this_fake_a.mean() - this_true_a.mean())
        ad_loss_b += (this_fake_b.mean() - this_true_b.mean())
        panalty_a, panalty_b = self.calc_gradient_penalty(in_a, fake_a, in_b, fake_b)
        ad_loss_a += panalty_a
        ad_loss_b += panalty_b

      
      #if it==0:
      #  ad_loss_a = ad_true_loss_a + ad_fake_loss_a
      #  ad_loss_b = ad_true_loss_b + ad_fake_loss_b
      #else:
      #  ad_loss_a += ad_true_loss_a + ad_fake_loss_a
      #  ad_loss_b += ad_true_loss_b + ad_fake_loss_b
      true_a_acc = _compute_true_acc(out_true_a)
      true_b_acc = _compute_true_acc(out_true_b)
      fake_a_acc = _compute_fake_acc(out_fake_a)
      fake_b_acc = _compute_fake_acc(out_fake_b)
      exec( 'self.dis_true_acc_%d = 0.5 * (true_a_acc + true_b_acc)' %it)
      exec( 'self.dis_fake_acc_%d = 0.5 * (fake_a_acc + fake_b_acc)' %it)


    d_loss = ( ad_loss_a + ad_loss_b )
    if 0 != hyperparameters['gamma_js_regularization']:
      d_loss += (js_reg_a + js_reg_b) * (hyperparameters['gamma_js_regularization']/2.)
    loss = hyperparameters['gan_w'] * d_loss
    self.dis.zero_grad()
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
    #pdb.set_trace()
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
    with torch.cuda.device(gpu):
        self.dis.cuda(gpu)
        self.gen.cuda(gpu)
        self.xy = self.xy.cuda(gpu)
    self.use_cuda = True

  def share_memory(self):
    self.dis.share_memory()
    self.gen.share_memory()
    #self.xy = self.xy.share_memory()


  def normalize_image(self, x):
    return x[:,0:3,:,:]

  def calc_gradient_penalty(self, real_data_a, fake_data_a, real_data_b, fake_data_b):
    # print "real_data: ", real_data.size(), fake_data.size()
    LAMBDA = 10
    alpha = torch.rand(1, 1)
    fake_data_a = fake_data_a.view(1, 3, 256, 256).data
    fake_data_b = fake_data_b.view(1, 3, 256, 256).data
    real_data_a = real_data_a.view(1, 3, 256, 256).data
    real_data_b = real_data_b.view(1, 3, 256, 256).data
    alpha = alpha.expand(1, real_data_a.nelement()/1).contiguous().view(1, 3, 256, 256)
    alpha = alpha.cuda(self.gpu) if self.use_cuda else alpha

    interpolates_a = alpha * real_data_a + ((1 - alpha) * fake_data_a)
    interpolates_b = alpha * real_data_b + ((1 - alpha) * fake_data_b)

    if self.use_cuda:
      interpolates_a = interpolates_a.cuda(self.gpu)
      interpolates_b = interpolates_b.cuda(self.gpu)
    interpolates_a = autograd.Variable(interpolates_a, requires_grad=True)
    interpolates_b = autograd.Variable(interpolates_b, requires_grad=True)

    disc_interpolates_a, disc_interpolates_b = self.dis(interpolates_a, interpolates_b)
    disc_interpolates_a, disc_interpolates_b = disc_interpolates_a[0], disc_interpolates_b[0]

    gradients_a = autograd.grad(outputs=disc_interpolates_a, inputs=interpolates_a,
                        grad_outputs=torch.ones(disc_interpolates_a.size()).cuda(self.gpu) if self.use_cuda else torch.ones(
                                                              disc_interpolates_a.size()),
                                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_b = autograd.grad(outputs=disc_interpolates_b, inputs=interpolates_b,
                        grad_outputs=torch.ones(disc_interpolates_b.size()).cuda(self.gpu) if self.use_cuda else torch.ones(
                                                              disc_interpolates_b.size()),
                                                      create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradients_a = gradients_a.view(gradients_a.size(0), -1)
    gradients_b = gradients_b.view(gradients_b.size(0), -1)

    gradient_penalty_a = ((gradients_a.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    gradient_penalty_b = ((gradients_b.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty_a,  gradient_penalty_b

  def js_regularization(self, D1_logits, D1_arg, D2_logits, D2_arg):
    D1 = nn.functional.sigmoid(D1_logits)
    D2 = nn.functional.sigmoid(D2_logits)
    grad_D1_logits = autograd.grad(outputs=D1_logits, inputs=D1_arg,
                                    grad_outputs=torch.ones(D1_logits.size()).cuda(self.gpu) if self.use_cuda else torch.ones(D1_logits.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_D2_logits = autograd.grad(outputs=D2_logits, inputs=D2_arg,
                                    grad_outputs=torch.ones(D2_logits.size()).cuda(self.gpu) if self.use_cuda else torch.ones(D2_logits.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    grad_D1_logits = grad_D1_logits.view(D1.size(0), -1)
    grad_D2_logits = grad_D2_logits.view(D2.size(0), -1)
    grad_D1_logits.norm(2, dim=1)
    grad_D2_logits.norm(2, dim=1)
    assert grad_D1_logits_norm.size() == D1.size()
    assert grad_D2_logits_norm.size() == D2.size()

    reg_D1 = torch.mul(torch.pow(1.0-D1, 2), torch.pow(grad_D1_logits_norm, 2))
    reg_D2 = torch.mul(torch.pow(D2, 2), torch.pow(grad_D2_logits_norm, 2))
    disc_regularizer = torch.mean(reg_D1 + reg_D2)
    return disc_regularizer

    
