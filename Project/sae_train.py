
#dictionary learning 
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader, random_split

from dictionary_learning.trainers.top_k import TopKTrainer, AutoEncoderTopK



DATA_PATH = "/data/saba/parnia/activations_by_t/t_0.8"
SAVE_DIR = Path("sae_output")

ACTIVATION_DIM = 192
DICT_SIZE = 192 * 8

K = 5

TRAINING_STEPS = 5000
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
EVAL_EVERY = 100

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)


#creating the data set from the .pt files we havee
class ActivationDataset(Dataset):
    def __init__(self, data_path):
        self.files = sorted(Path(data_path).glob("*.pt"))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = torch.load(self.files[idx],map_location="cpu")
        activation = data["activation"].float()
    
        return activation


dataset = ActivationDataset(DATA_PATH)

print("number of files", len(dataset))
# print("shape:", dataset[0].shape)



n_train = int(len(dataset) * 0.8)
n_val = len(dataset) - n_train

generator = torch.Generator().manual_seed(42)
train_dataset, val_dataset = random_split(dataset,[n_train, n_val],generator=generator)

print("\ntrain samples:", len(train_dataset))
print("validation samples:", len(val_dataset))



train_loader = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
val_loader = DataLoader(val_dataset,batch_size=BATCH_SIZE,shuffle=False)

print("train batches:", len(train_loader))
print("validation batches:", len(val_loader))



trainer_cfg = {
    "dict_class": AutoEncoderTopK,

    "activation_dim": ACTIVATION_DIM,
    "dict_size": DICT_SIZE,

    "k": K,

    "lr": LEARNING_RATE,

    "device": device,
    "steps": TRAINING_STEPS,

    # required metadata
    "layer": 1,
    "lm_name": "diffusion_planner",

    "warmup_steps": 50,
    # Start learning inference threshold earlier because
    # this is only a 1000-step training run.
    # "threshold_start_step": 100,
}


trainer = TopKTrainer(**trainer_cfg)
model = trainer.ae

@torch.no_grad()
def evaluate(trainer,loader):
    trainer.ae.eval() #the actual SAE
    total_squared_error = 0.0
    total_vectors = 0
    total_elements = 0
    
    for acts in loader:
        # print(acts.shape)
        acts = acts.to(device).reshape(-1, ACTIVATION_DIM)

        recon = trainer.ae(acts)

        squared_error = (recon - acts).pow(2)
        total_squared_error += squared_error.sum().item()
        total_vectors += acts.shape[0]
        total_elements += acts.numel() #number of elemnts

    
    l2 = (total_squared_error / total_vectors) ** 0.5

    return {"mse": l2}
    # return total_squared_error / total_elements
    ...




step = 0
while step < TRAINING_STEPS:
    for acts in train_loader:
        if step >= TRAINING_STEPS:
            break

        acts = acts.to(device).reshape(-1, ACTIVATION_DIM)
        train_loss = trainer.update(step, acts)  # train_loss = mse_loss + l1_penalty * sparsity_loss


        if step % EVAL_EVERY == 0:
            metrics = evaluate(trainer, val_loader)
            print(
                f"step {step:4d} | "
                f"train loss {train_loss:.6f} | "
                f"val MSE {metrics['mse']:.8f}"
            )

        step += 1



# Save SAE
SAVE_DIR.mkdir(parents=True, exist_ok=True)
save_path = SAVE_DIR / "ae_final_t8.pt"
torch.save(trainer.ae.state_dict(), save_path)

print("\nTraining complete!")
print("SAE saved to:", save_path)



# for t 1 :

'''
step  100 | train loss 295.713257 | val MSE 14.81397767
step  200 | train loss 123.365425 | val MSE 11.97171513
step  300 | train loss 100.896599 | val MSE 10.57193033
step  400 | train loss 74.876480 | val MSE 9.63338865
step  500 | train loss 61.800011 | val MSE 8.95714127
step  600 | train loss 45.754852 | val MSE 8.61052192
step  700 | train loss 61.882225 | val MSE 8.18415889
step  800 | train loss 42.732578 | val MSE 7.84905539
step  900 | train loss 35.908691 | val MSE 7.65688991
step 1000 | train loss 47.909470 | val MSE 7.51170114
step 1100 | train loss 35.972279 | val MSE 7.27271530
step 1200 | train loss 47.138161 | val MSE 7.16148110
step 1300 | train loss 32.487770 | val MSE 6.97954949
step 1400 | train loss 20.794865 | val MSE 6.95517807
step 1500 | train loss 40.617580 | val MSE 6.84640813
step 1600 | train loss 24.386883 | val MSE 6.79298190
step 1700 | train loss 40.458267 | val MSE 6.66854144
step 1800 | train loss 30.304134 | val MSE 6.66176540
step 1900 | train loss 34.037922 | val MSE 6.54672588
step 2000 | train loss 25.576134 | val MSE 6.47467190
step 2100 | train loss 34.849674 | val MSE 6.42392947
step 2200 | train loss 16.912123 | val MSE 6.37529427
step 2300 | train loss 26.068798 | val MSE 6.37049521
step 2400 | train loss 34.711796 | val MSE 6.32601162
step 2500 | train loss 50.289906 | val MSE 6.26110371
step 2600 | train loss 23.304106 | val MSE 6.25977574
step 2700 | train loss 17.170612 | val MSE 6.22524869
step 2800 | train loss 16.471344 | val MSE 6.16270774
step 2900 | train loss 24.904455 | val MSE 6.14901328
step 3000 | train loss 17.492893 | val MSE 6.15300278
step 3100 | train loss 16.056200 | val MSE 6.09105636
step 3200 | train loss 19.732479 | val MSE 6.07379368
step 3300 | train loss 23.835638 | val MSE 6.01350559
step 3400 | train loss 16.508224 | val MSE 6.05463209
step 3500 | train loss 18.526371 | val MSE 6.07841487
step 3600 | train loss 16.379904 | val MSE 5.96297008
step 3700 | train loss 22.055689 | val MSE 6.06919016
step 3800 | train loss 18.010138 | val MSE 5.98291649
step 3900 | train loss 20.006432 | val MSE 6.03387582
step 4000 | train loss 16.277008 | val MSE 5.95813221
step 4100 | train loss 32.753815 | val MSE 5.95152297
step 4200 | train loss 16.270248 | val MSE 6.03634710
step 4300 | train loss 15.894505 | val MSE 5.95601822
step 4400 | train loss 16.193260 | val MSE 5.96782586
step 4500 | train loss 15.436737 | val MSE 5.95109570
step 4600 | train loss 17.852003 | val MSE 5.99119088
step 4700 | train loss 20.436867 | val MSE 5.88868134
step 4800 | train loss 26.099112 | val MSE 5.98049487
step 4900 | train loss 11.314738 | val MSE 5.92263845

Training complete!
SAE saved to: sae_output/ae_final_t1.pt
'''

# for t = 0.2 :

'''step    0 | train loss 283.653625 | val MSE 19.48840764
step  100 | train loss 208.003540 | val MSE 14.71975408
step  200 | train loss 141.358719 | val MSE 11.45550178
step  300 | train loss 80.454582 | val MSE 10.02086191
step  400 | train loss 59.355614 | val MSE 9.19369702
step  500 | train loss 64.642220 | val MSE 8.70300690
step  600 | train loss 51.490379 | val MSE 8.25413762
step  700 | train loss 53.971245 | val MSE 7.83952866
step  800 | train loss 47.808594 | val MSE 7.62526676
step  900 | train loss 32.402138 | val MSE 7.36429880
step 1000 | train loss 40.804993 | val MSE 7.23273005
step 1100 | train loss 30.199860 | val MSE 7.08208258
step 1200 | train loss 39.462440 | val MSE 6.92766801
step 1300 | train loss 36.878601 | val MSE 6.73212068
step 1400 | train loss 31.689180 | val MSE 6.64610212
step 1500 | train loss 39.544304 | val MSE 6.60527825
step 1600 | train loss 22.167393 | val MSE 6.47473714
step 1700 | train loss 25.297335 | val MSE 6.43983177
step 1800 | train loss 31.808105 | val MSE 6.29660244
step 1900 | train loss 31.890800 | val MSE 6.21974725
step 2000 | train loss 19.904715 | val MSE 6.13747417
step 2100 | train loss 20.158503 | val MSE 6.05685933
step 2200 | train loss 14.949183 | val MSE 6.01758887
step 2300 | train loss 24.552099 | val MSE 5.96695626
step 2400 | train loss 22.161964 | val MSE 5.92815759
step 2500 | train loss 26.108624 | val MSE 5.87953930
step 2600 | train loss 23.926594 | val MSE 5.90972989
step 2700 | train loss 23.920750 | val MSE 5.79108590
step 2800 | train loss 31.750463 | val MSE 5.80258197
step 2900 | train loss 17.806374 | val MSE 5.75432186
step 3000 | train loss 15.169048 | val MSE 5.72276913
step 3100 | train loss 12.057355 | val MSE 5.69452437
step 3200 | train loss 19.236767 | val MSE 5.70466175
step 3300 | train loss 33.396591 | val MSE 5.62878746
step 3400 | train loss 16.184689 | val MSE 5.63301256
step 3500 | train loss 24.925095 | val MSE 5.62570228
step 3600 | train loss 17.550621 | val MSE 5.65663064
step 3700 | train loss 18.783037 | val MSE 5.56792233
step 3800 | train loss 22.821548 | val MSE 5.57384081
step 3900 | train loss 12.792994 | val MSE 5.53510999
step 4000 | train loss 15.080427 | val MSE 5.56314827
step 4100 | train loss 30.903841 | val MSE 5.59213536
step 4200 | train loss 10.495125 | val MSE 5.52701758
step 4300 | train loss 16.899870 | val MSE 5.53326855
step 4400 | train loss 15.297272 | val MSE 5.54580481
step 4500 | train loss 17.614910 | val MSE 5.52008199
step 4600 | train loss 16.844929 | val MSE 5.51839105
step 4700 | train loss 23.107494 | val MSE 5.44518434
step 4800 | train loss 17.038708 | val MSE 5.50332409
step 4900 | train loss 13.482163 | val MSE 5.50026465

Training complete!'''





