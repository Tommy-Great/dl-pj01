import matplotlib.pyplot as plt


def plot_training_loss_acc(runner, sample_step=20):
    fig_size = (12, 6)
    loss_legend_loc = "upper right"
    acc_legend_loc = "lower right"
    train_color = "deeppink"
    dev_color = 'lightpink'
    font_size = 'large'
    train_linestyle = '-'
    dev_linestyle = '--'

    plt.figure(figsize=fig_size)

    plt.subplot(1, 2, 1)
    train_steps = list(range(1, len(runner.step_losses) + 1, sample_step))
    train_losses = [float(runner.step_losses[step_ - 1]) for step_ in train_steps]
    plt.plot(train_steps, train_losses, color=train_color, linestyle=train_linestyle, label='Train loss')
    if len(runner.dev_losses) > 0:
        dev_steps = [item[0] for item in runner.dev_losses]
        dev_losses = [float(item[1]) for item in runner.dev_losses]
        plt.plot(dev_steps, dev_losses, color=dev_color, linestyle=dev_linestyle, label='dev loss')
    plt.ylabel('loss', fontsize=font_size)
    plt.xlabel('step', fontsize=font_size)
    plt.legend(loc=loss_legend_loc, fontsize='x-large')

    if len(runner.dev_scores) > 0:
        plt.subplot(1, 2, 2)
        dev_steps = [item[0] for item in runner.dev_scores]
        dev_scores = [float(item[1]) for item in runner.dev_scores]
        plt.plot(dev_steps, dev_scores, color=dev_color, linestyle=dev_linestyle, label='dev score')
        plt.ylabel('score', fontsize=font_size)
        plt.xlabel('step', fontsize=font_size)
        plt.legend(loc=acc_legend_loc, fontsize='x-large')
    plt.savefig('./fig01.png')
    print('fig saved !')
    plt.show()  # 这个似乎会引发线程错误,之后调用train_loader()可能会报错
