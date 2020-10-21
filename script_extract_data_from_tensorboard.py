import tensorboardX
import matplotlib.pyplot as plt
import os

def load_tensorboard_csv(f):
    l = f.readline() # first line is walltime, step, value
    l = f.readline()
    step = []
    value = []
    while len(l) > 0:
        l = l[:-1]
        v = l.split(',')
        step.append(int(v[1]))
        value.append(float(v[2]))
        l = f.readline()[1:-1]
    return step, value


def main():
    # export tensorboard data to logging_csv before running
    logfiles = ['Median', 'loss']
    splits = ['train', 'test']
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Median error', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    ax2.set_ylim(-6, 2)
    all_plots = []
    for split, linetype in zip(splits, ('', '--')):
        for logfile, color,ax in zip(logfiles, ('r', 'b'), (ax1, ax2)):
            filename = os.path.join('logging_csv', split, '{}.csv'.format(logfile))
            with open(filename, 'r') as f:
                step, value = load_tensorboard_csv(f)
            new_plot, = ax.plot(step, value, linetype+color, label='{} {}'.format(split, logfile))
            all_plots.append(new_plot)
    ax1.legend(handles=all_plots, loc='best')
    plt.savefig('plots/Modelnet_loss_overfit.pdf')



if __name__ == '__main__':
    main()
