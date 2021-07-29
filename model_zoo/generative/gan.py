import hydra
import torch

from torch.nn import functional as F

from upcycle import cuda


class GAN(torch.nn.Module):
    def __init__(self, latent_size, output_size, gen_net, disc_net):
        super().__init__()
        self.generator = gen_net
        self.discriminator = disc_net
        self.latent_size = latent_size

    def gen_backward(self, batch_size):
        # Generator hinge loss
        fake_samples = self.sample(batch_size)
        fake_logits = self.discriminator(fake_samples)
        loss = -torch.mean(fake_logits)
        loss.backward()
        return loss, fake_samples

    def disc_backward(self, real_samples):
        # Discriminator hinge loss
        batch_size = real_samples.size(0)
        with torch.no_grad():
            fake_samples = self.sample(batch_size)
        real_logits = self.discriminator(real_samples)
        fake_logits = self.discriminator(fake_samples)
        loss = F.relu(1 - real_logits).mean() + F.relu(1 + fake_logits).mean()
        loss.backward()

        with torch.no_grad():
            real_preds = (1 - real_logits) < 0.
            false_neg_rate = real_preds.float().mean()  # num. of real samples identified as fake
            fake_preds = (1 + fake_logits) < 0.
            false_pos_rate = fake_preds.float().mean()  # num. of fake samples identified as real

        return loss, false_neg_rate, false_pos_rate

    def sample(self, batch_size):
        latent_samples = torch.normal(mean=0., std=1., size=(batch_size, self.latent_size))
        fake_samples = self.generator(latent_samples)
        return fake_samples

    def gan_train_epoch(self, gen_opt, disc_opt, gen_lr_sched, disc_lr_sched,
                        train_loader, test_loader, train_cfg):
        num_batches = len(train_loader)
        self.train()
        gen_update_count = 0
        while gen_update_count < train_cfg.eval_period:
            tot_gen_loss = tot_disc_loss = 0
            tot_false_neg = tot_false_pos = 0

            for i, (real_samples, _) in enumerate(train_loader):
                real_samples = cuda.try_cuda(real_samples)
                batch_size = real_samples.size(0)

                if i % train_cfg.gen_update_period == 0:
                    gen_opt.zero_grad()
                    gen_loss, _ = self.gen_backward(batch_size)
                    gen_opt.step()
                    gen_update_count += 1

                if i % train_cfg.disc_update_period == 0:
                    disc_opt.zero_grad()
                    disc_loss, false_neg, false_pos = self.disc_backward(real_samples)
                    disc_opt.step()

                # step lr here, assuming decay is pegged to gen_update_count
                if i % train_cfg.gen_update_period == 0:
                    gen_lr_sched.step()
                    disc_lr_sched.step()

                # logging
                tot_gen_loss += gen_loss.item() / num_batches
                tot_disc_loss += disc_loss.item() / num_batches
                tot_false_neg += false_neg.item() / num_batches
                tot_false_pos += false_pos.item() / num_batches

                if gen_update_count == train_cfg.eval_period:
                    break
        self.eval()

        metrics = dict(gen_loss=tot_gen_loss, disc_loss=tot_disc_loss,
                       false_neg=tot_false_neg, false_pos=tot_false_pos)

        fid_score = is_score = float('NaN')
        metrics.update(dict(fid_score=float(fid_score), is_score=float(is_score)))

        return metrics

    # def fit(self):
    #     decay_fn = model_zoo.utils.training.linear_decay_handle(config.trainer.optimizer.lr,
    #                                                             config.trainer.lr_decay.min_lr,
    #                                                             config.trainer.lr_decay.start,
    #                                                             config.trainer.lr_decay.stop)
    #     gen_lr_sched = torch.optim.lr_scheduler.LambdaLR(gen_opt, lr_lambda=decay_fn)
    #     disc_lr_sched = torch.optim.lr_scheduler.LambdaLR(disc_opt, lr_lambda=decay_fn)
    #
    #     logger.add_table('train_metrics')
    #     gen_update_count = 0
    #     while gen_update_count < config.trainer.num_gen_updates:
    #         metrics = gan_train_epoch(density_model, gen_opt, disc_opt, gen_lr_sched, disc_lr_sched,
    #                                   train_loader, None, config.trainer)
    #         gen_update_count += config.trainer.eval_period
    #         last_lr = gen_lr_sched.get_last_lr()[0]
    #
    #         print(f'[GAN] : step {gen_update_count}, lr: {last_lr:.6f}, ' \
    #               f'gen loss: {metrics["gen_loss"]:0.4f}, disc loss: {metrics["disc_loss"]:0.4f}, ' \
    #               f'false neg: {metrics["false_neg"]:.4f}, false pos: {metrics["false_pos"]:.4f}')
    #         logger.log(metrics, gen_update_count, 'train_metrics')
    #         logger.write_csv()
    #         if gen_update_count == config.trainer.checkpoint_period:
    #             logger.save_obj(density_model.generator.state_dict(),
    #                             f'{task}_generator_{gen_update_count}.pkl')
    #             logger.save_obj(density_model.discriminator.state_dict(),
    #                             f'{task}_discriminator_{gen_update_count}.pkl')


class RegressionBoundaryGAN(GAN):
    def __init__(self, latent_size, output_size, gen_net, disc_net, cond_model, prior_weight):
        super().__init__(latent_size, output_size, gen_net, disc_net)
        self.cond_model = cond_model
        self.prior_weight = prior_weight

    def fit_cond_model(self, *args, **kwargs):
        cond_metrics = self.cond_model.fit(*args, **kwargs)
        return cond_metrics

    def gen_backward(self, batch_size):
        gan_loss, fake_samples = super().gen_backward(batch_size)

        cond_mean, cond_var = self.cond_model.predict(fake_samples, compat_mode='torch')
        cond_dist = torch.distributions.Normal(cond_mean, cond_var.sqrt())

        prior_mean = torch.zeros_like(cond_mean)
        prior_std = torch.ones_like(cond_var)
        prior_dist = torch.distributions.Normal(prior_mean, prior_std)
        prior_loss = torch.distributions.kl_divergence(prior_dist, cond_dist).mean()

        loss = gan_loss + self.prior_weight * prior_loss

        return loss, fake_samples
