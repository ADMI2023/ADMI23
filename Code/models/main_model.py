import numpy as np
import torch
import torch.nn as nn
from models.diffusion import diff_ADMI

class CSDI_base(nn.Module):
    def __init__(self, configs, pretrain_model):
        super().__init__()
        self.configs = configs
        self.device = configs.device
        self.target_dim = configs.enc_in

        self.emb_time_dim = configs.timeemb
        self.emb_feature_dim = configs.featureemb
        self.target_strategy = configs.target_strategy

        self.emb_total_dim = self.emb_time_dim + self.emb_feature_dim
        self.configs.side_dim = self.emb_total_dim
        self.embed_layer = nn.Embedding(
            num_embeddings=self.configs.enc_in, embedding_dim=self.emb_feature_dim
        )

        self.diffmodel = diff_ADMI(self.configs, pretrain_model)

        # parameters for diffusion models
        self.num_steps = configs.diffusion_step_num
        if configs.schedule == "quad":
            self.beta = np.linspace(
                configs.beta_start ** 0.5, configs.beta_end ** 0.5, self.num_steps
            ) ** 2
        elif configs.schedule == "linear":
            self.beta = np.linspace(
                configs.beta_start, configs.beta_end, self.num_steps
            )
        self.alpha_hat = 1 - self.beta
        self.alpha = np.cumprod(self.alpha_hat)
        self.alpha_torch = torch.tensor(self.alpha).float().to(self.device).unsqueeze(1).unsqueeze(1)

    def process_data(self, observed_data, observed_mask,observed_tp,gt_mask):
        observed_data = observed_data.to(self.device).float().permute(0,2,1)
        observed_mask = observed_mask.to(self.device).float().permute(0,2,1)
        observed_tp = observed_tp.to(self.device).float()
        gt_mask = gt_mask.to(self.device).float().permute(0,2,1)

        return (observed_data, observed_mask, observed_tp, gt_mask)
    
    def time_embedding(self, pos, d_model=128):
        pe = torch.zeros(pos.shape[0], pos.shape[1], d_model).to(self.device)
        position = pos.unsqueeze(2)
        div_term = 1 / torch.pow(
            10000.0, torch.arange(0, d_model, 2).to(self.device) / d_model
        )
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe

    def get_speedmask(self, observed_data, observed_mask):
        cond_mask = observed_mask.clone()
        B, K, L = observed_data.shape
   
        for sample_id in range(B):
            data = observed_data[sample_id].detach().to("cpu").numpy()
            speed = np.abs(np.diff(data, prepend=0.001, axis = 1)) #[K, L]
            speed = speed / self.configs.temperature
            e_s = np.exp(speed - np.max(speed, axis=1).reshape(K, 1))
            probability = e_s / e_s.sum(axis = 1).reshape(K, 1)
            
            sample_ratio = np.random.rand()
            for attr in range(K):
                choice_num = int(L * sample_ratio / self.configs.avg_mask_len_ssl)
                mask_indexs = np.random.choice(L, choice_num, p = probability[attr].ravel())
                mask_lens = np.random.geometric(1 / self.configs.avg_mask_len_ssl, choice_num)
                for i in range(choice_num):
                    start = max(0, mask_indexs[i] - int( mask_lens[i] / 2))
                    end = min(data.shape[0],  mask_indexs[i] + int( mask_lens[i] / 2) + 1)
                    cond_mask[sample_id, attr, start:end] = 0
        return cond_mask
    
    def get_randmask(self, observed_mask):
        rand_for_mask = torch.rand_like(observed_mask) * observed_mask
        rand_for_mask = rand_for_mask.reshape(len(rand_for_mask), -1)
        for i in range(len(observed_mask)):
            sample_ratio = np.random.rand()  # missing ratio
            num_observed = observed_mask[i].sum().item()
            num_masked = round(num_observed * sample_ratio)
            rand_for_mask[i][rand_for_mask[i].topk(num_masked).indices] = -1
        cond_mask = (rand_for_mask > 0).reshape(observed_mask.shape).float()
        return cond_mask

    def get_hist_mask(self, observed_mask):
        for_pattern_mask = observed_mask
        if self.target_strategy == "mix":
            rand_mask = self.get_randmask(observed_mask)

        cond_mask = observed_mask.clone()
        for i in range(len(cond_mask)):
            mask_choice = np.random.rand()
            if self.target_strategy == "mix" and mask_choice > 0.5:
                cond_mask[i] = rand_mask[i]
            else:  # draw another sample for histmask (i-1 corresponds to another sample)
                cond_mask[i] = cond_mask[i] * for_pattern_mask[i - 1] 
        return cond_mask

    def get_side_info(self, observed_tp, cond_mask):
        B, K, L = cond_mask.shape

        time_embed = self.time_embedding(observed_tp, self.emb_time_dim)  # (B,L,emb)
        time_embed = time_embed.unsqueeze(2).expand(-1, -1, K, -1)
        feature_embed = self.embed_layer(
            torch.arange(self.target_dim).to(self.device)
        )  # (K,emb)
        feature_embed = feature_embed.unsqueeze(0).unsqueeze(0).expand(B, L, -1, -1)
        side_info = torch.cat([time_embed, feature_embed], dim=-1)  # (B,L,K,*)
        side_info = side_info.permute(0, 3, 2, 1)  # (B,*,K,L)

        return side_info

    def set_input_to_diffmodel(self, noisy_data, observed_data, cond_mask):
        B, K, L = observed_data.shape

        cond_obs = (cond_mask * observed_data).unsqueeze(1)
        noisy_target = ((1 - cond_mask) * noisy_data).unsqueeze(1)
        total_input = torch.cat([cond_obs, noisy_target], dim=1)  # (B,2,K,L)
        return total_input

    def calc_loss(self, observed_data, cond_mask, observed_mask, side_info):
        B, K, L = observed_data.shape
        #Generate Time Step
        diffusion_step = torch.randint(0, self.num_steps, [B]).to(self.device)
        current_alpha = self.alpha_torch[diffusion_step]  # (B,1,1)
        noise = torch.randn_like(observed_data)
        noisy_data = (current_alpha ** 0.5) * observed_data + (1.0 - current_alpha) ** 0.5 * noise
        
        cond_obs = cond_mask * observed_data
        total_input = self.set_input_to_diffmodel(noisy_data, observed_data, cond_mask)

        predicted = self.diffmodel(total_input, side_info, cond_obs, diffusion_step)  # (B,K,L)

        target_mask = observed_mask - cond_mask
        residual = (noise - predicted) * target_mask
        num_eval = target_mask.sum()
        loss = (residual ** 2).sum() / (num_eval if num_eval > 0 else 1)
        return loss

    def impute(self, observed_data, cond_mask, side_info, n_samples=100):
        B, K, L = observed_data.shape

        imputed_samples = torch.zeros(B, n_samples, K, L).to(self.device)

        for i in range(n_samples):
            current_sample = torch.randn_like(observed_data)

            for t in range(self.num_steps - 1, -1, -1):
                cond_obs = (cond_mask * observed_data)
                total_input = self.set_input_to_diffmodel(current_sample, observed_data, cond_mask)
                predicted = self.diffmodel(total_input, side_info, cond_obs, torch.tensor([t]).to(self.device), mode="TEST")

                coeff1 = 1 / self.alpha_hat[t] ** 0.5
                coeff2 = (1 - self.alpha_hat[t]) / (1 - self.alpha[t]) ** 0.5
                current_sample = coeff1 * (current_sample - coeff2 * predicted)

                if t > 0:
                    noise = torch.randn_like(current_sample)
                    sigma = (
                        (1.0 - self.alpha[t - 1]) / (1.0 - self.alpha[t]) * self.beta[t]
                    ) ** 0.5
                    current_sample += sigma * noise

            imputed_samples[:, i] = current_sample.detach()
        return imputed_samples

    def forward(self, observed_data, observed_mask, observed_tp, gt_mask):
        (observed_data, observed_mask, observed_tp, gt_mask) = self.process_data(observed_data, observed_mask, observed_tp, gt_mask)
        observed_data *= observed_mask
        if self.target_strategy != "random":
            cond_mask = self.get_speedmask(observed_data, observed_mask)
        else:
            cond_mask = self.get_randmask(observed_mask)

        side_info = self.get_side_info(observed_tp, cond_mask)

        return self.calc_loss(observed_data, cond_mask, observed_mask, side_info)
    
    def evaluate(self, observed_data, observed_mask, observed_tp, gt_mask, n_samples=100):
        (observed_data, observed_mask, observed_tp, gt_mask) = self.process_data(observed_data, observed_mask, observed_tp, gt_mask)
        observed_data *= observed_mask
        with torch.no_grad():
            cond_mask = observed_mask
            side_info = self.get_side_info(observed_tp, cond_mask)
            imputed_samples = self.impute(observed_data, cond_mask, side_info, n_samples)
            return imputed_samples