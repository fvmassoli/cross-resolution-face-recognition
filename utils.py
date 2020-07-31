import os
import torch


def load_models(model_base_path, device="cpu", model_ckp=None):
    assert os.path.exists(model_base_path), f"Base model checkpoint not found at: {model_base_path}"
    sm = torch.load(model_base_path)
    tm = torch.load(model_base_path)
    if model_ckp is not None:
        assert os.path.exists(model_ckp), f"Model checkpoint not found at: {model_ckp}"
        ckp = torch.load(self._fine_tuned_model_path, map_location='cpu')
        [p.data.copy_(torch.from_numpy(ckp['model_state_dict'][n].numpy())) for n, p in model.named_parameters()]
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.momentum = 0.1
                m.running_var = ckp['model_state_dict'][n + '.running_var']
                m.running_mean = ckp['model_state_dict'][n + '.running_mean']
                m.num_batches_tracked = ckp['model_state_dict'][n + '.num_batches_tracked']
    ## Freeze all params for the teacher model
    for param in tm.parameters():
        param.requires_grad = False
    return sm.to(device), tm.to(device)


def save_model_checkpoint(best_acc, batch_idx, epoch, model_state_dict, out_dir, logging):
    state_dict = {
                'best_acc': best_acc,
                'epoch': epoch,
                'model_state_dict': model_state_dict
            }
    file_name = os.path.join(out_dir, f'models_ckp_{epoch}_{batch_idx}.pth')
    torch.save(state_dict, file_name)
    logging.info(
            f"Saved model with best acc: {best_acc}"
            f"\nAt epoch: {epoch}"
            f"\nAt iter: {batch_idx}"
            f"\nModel saved at: {file_name}"
        )
