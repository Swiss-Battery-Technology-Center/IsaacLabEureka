def compute_success_metric(self, env_ids):
    success_rate = self.episode_length_buf[:].float().mean() / self.max_episode_length
    return {"success_metric": success_rate,}