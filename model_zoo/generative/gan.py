import hydra
import torch

from torch.nn import functional as F


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
