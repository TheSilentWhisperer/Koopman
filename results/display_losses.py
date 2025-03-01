import json
import os
import matplotlib.pyplot as plt

def get_losses(path, eval = True):

    with open(path, "r") as f:
        data = json.load(f)

    if not eval:
        return data
    
    # data["train"] = [x ** 0.5 for x in data["train"]]
    # data["eval"] = [x ** 0.5 for x in data["eval"]]
    # return data["train"], data["eval"]
    return data

def plot_loss(ax, key, title, eval = True):
    filename = filenames[key]
    if eval:
        train_losses, eval_losses = get_losses(filename)
        ax.plot(train_losses, label="train")
        ax.plot(eval_losses, label="eval")
        ax.legend()
    else:
        losses = get_losses(filename, eval)
        #plot the losses with x-axis scaled to seconds
        ax.plot([i * 0.05 for i in range(len(losses))], losses, label="eval")
    #ax.set_title(title)


filenames = {
    # "autoencoder_pretraining": "autoencoder_pretraining/losses.json",
    "full_model": "data/full_model_losses.json"
}

if __name__ == "__main__":
    fig, axs = plt.subplots(1,1)
    plot_loss(axs, "full_model", "Full Model", eval = False)
    axs.set_xlabel("time (s)")
    axs.set_ylabel("Average pixel RMSE")
    #set y to go from 0 to 1
    axs.set_ylim([0.005, 0.064])
    plt.show()