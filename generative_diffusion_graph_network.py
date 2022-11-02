import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np


class ConcatSquashLinear(tf.keras.Model):
    def __init__(self, dim_in, dim_out, dim_ctx):
        super(ConcatSquashLinear, self).__init__()
        self._layer = tf.keras.layers.Dense(dim_in, dim_out)
        self._hyper_bias = tf.keras.layers.Dense(dim_ctx, dim_out, bias=False)
        self._hyper_gate = tf.keras.layers.Dense(dim_ctx, dim_out)

    def forward(self, ctx, x):
        gate = tf.sigmoid(self._hyper_gate(ctx))
        bias = self._hyper_bias(ctx)
        # if x.dim() == 3:
        #     gate = gate.unsqueeze(1)
        #     bias = bias.unsqueeze(1)
        ret = self._layer(x) * gate + bias
        return ret


class VarianceSchedule(tf.keras.Model):
    def __init__(self, num_steps, beta_1, beta_T, mode = 'linear'):
        super().__init__()
        assert mode in ('linear',)
        self.num_steps = num_steps
        self.beta_1 = beta_1
        self.beta_T = beta_T
        self.mode = mode

        if mode == 'linear':
            betas = tf.linspace(beta_1, beta_T, steps = num_steps)

        betas = tf.concat([tf.zeros([1]), betas], dim = 0)  # Padding

        alphas = 1 - betas
        log_alphas = tf.math.log(alphas)
        for i in range(1, log_alphas.size(0)):  # 1 to T
            log_alphas[i] += log_alphas[i - 1]
        alpha_bars = log_alphas.exp()

        sigmas_flex = tf.math.sqrt(betas)
        sigmas_inflex = tf.zeros_like(sigmas_flex)
        for i in range(1, sigmas_flex.size(0)):
            sigmas_inflex[i] = ((1 - alpha_bars[i - 1]) / (1 - alpha_bars[i])) * betas[i]
        sigmas_inflex = tf.math.sqrt(sigmas_inflex)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('sigmas_flex', sigmas_flex)
        self.register_buffer('sigmas_inflex', sigmas_inflex)

    def uniform_sample_t(self, batch_size):
        ts = np.random.choice(np.arange(1, self.num_steps + 1), batch_size)
        return ts.tolist()

    def get_sigmas(self, t, flexibility):
        assert 0 <= flexibility and flexibility <= 1
        sigmas = self.sigmas_flex[t] * flexibility + self.sigmas_inflex[t] * (1 - flexibility)
        return sigmas


class PointwiseNet(tf.keras.Model):
    def __init__(self, point_dim, context_dim, residual):
        super().__init__()
        self.act = tf.keras.layers.LeakyReLU
        self.residual = residual
        self.layers = [
            ConcatSquashLinear(3, 128, context_dim + 3),
            ConcatSquashLinear(128, 256, context_dim + 3),
            ConcatSquashLinear(256, 512, context_dim + 3),
            ConcatSquashLinear(512, 256, context_dim + 3),
            ConcatSquashLinear(256, 128, context_dim + 3),
            ConcatSquashLinear(128, 3, context_dim + 3)
        ]

    def forward(self, x, beta, context):
        """
        Args:
            x:  Point clouds at some timestep t, (B, N, d).
            beta:     Time. (B, ).
            context:  Shape latents. (B, F).
        """
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)  # (B, 1, 1)
        context = context.view(batch_size, 1, -1)  # (B, 1, F)

        time_emb = tf.concat([beta, tf.math.sin(beta), tf.math.cos(beta)], dim = -1)  # (B, 1, 3)
        ctx_emb = tf.concat([time_emb, context], dim = -1)  # (B, 1, F+3)

        out = x
        for i in range(len(self.layers)):
            layer = self.layers[i]
            out = layer(ctx = ctx_emb, x = out)
            if i < len(self.layers) - 1:
                out = self.act(out)

        if self.residual:
            return x + out
        else:
            return out


class DiffusionPoint(tf.keras.Model):
    def __init__(self, net, var_sched: VarianceSchedule):
        super().__init__()
        self.net = net
        self.var_sched = var_sched
        self.mse = tf.keras.losses.MeanSquaredError(reduction = tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)

    def get_loss(self, x_0, context, t = None):
        """
        Args:
            x_0:  Input point cloud, (B, N, d).
            context:  Shape latent, (B, F).
        """
        batch_size, _, point_dim = x_0.size()
        if t is None:
            t = self.var_sched.uniform_sample_t(batch_size)
        alpha_bar = self.var_sched.alpha_bars[t]
        beta = self.var_sched.betas[t]

        c0 = tf.math.sqrt(alpha_bar).view(-1, 1, 1)  # (B, 1, 1)
        c1 = tf.math.sqrt(1 - alpha_bar).view(-1, 1, 1)  # (B, 1, 1)

        e_rand = tf.random.normal(x_0)  # (B, N, d)
        e_theta = self.net(c0 * x_0 + c1 * e_rand, beta = beta, context = context)

        loss = self.mse(e_theta.view(-1, point_dim), e_rand.view(-1, point_dim))
        return loss

    def sample(self, num_points, context, point_dim = 3, flexibility = 0.0, ret_traj = False):
        batch_size = context.size(0)
        x_T = tf.random.normal([batch_size, num_points, point_dim]).to(context.device)
        traj = {self.var_sched.num_steps: x_T}
        for t in range(self.var_sched.num_steps, 0, -1):
            z = tf.random.normal(x_T) if t > 1 else tf.zeros_like(x_T)
            alpha = self.var_sched.alphas[t]
            alpha_bar = self.var_sched.alpha_bars[t]
            sigma = self.var_sched.get_sigmas(t, flexibility)

            c0 = 1.0 / tf.math.sqrt(alpha)
            c1 = (1 - alpha) / tf.math.sqrt(1 - alpha_bar)

            x_t = traj[t]
            beta = self.var_sched.betas[[t] * batch_size]
            e_theta = self.net(x_t, beta = beta, context = context)
            x_next = c0 * (x_t - c1 * e_theta) + sigma * z
            traj[t - 1] = x_next.detach()  # Stop gradient and save trajectory.
            traj[t] = traj[t].cpu()  # Move previous output to CPU memory.
            if not ret_traj:
                del traj[t]

        if ret_traj:
            return traj
        else:
            return traj[0]

