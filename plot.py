from plot.plot import plot_loss, plot_lr

log_path = "./log/train.log"
# plot_lr(log_path, save_path="lr.png")
plot_loss(log_path, save_path="loss.png")


