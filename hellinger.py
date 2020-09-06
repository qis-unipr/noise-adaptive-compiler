import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt


def plot(results_file, metric):
    with open(results_file, 'rb') as f:
        results = pkl.load(f)

    labels = [circ for circ in results]

    configs = list(results[labels[0]].keys())

    data = [[results[circ][config][metric] for circ in labels] for config in configs]

    x = 2 * np.arange(len(results))  # the label locations
    width = 0.25  # the width of the bars

    fig, ax = plt.subplots()
    rects = list()
    for i, series in enumerate(data):
        rects.append(ax.bar(x + i * width, series, width, label=configs[i]))

    # Add some text for labels, title and custom x-axis tick labels, etc.
    # ax.set_title('Scores by group and gender')
    ax.set_ylabel(metric)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # def autolabel(rects):
    #     """Attach a text label above each bar in *rects*, displaying its height."""
    #     for rect in rects:
    #         height = rect.get_height()
    #         ax.annotate('{}'.format(height),
    #                     xy=(rect.get_x() + rect.get_width() / 2, height),
    #                     xytext=(0, 3),  # 3 points vertical offset
    #                     textcoords="offset points",
    #                     ha='center', va='bottom')
    #
    #
    # for rect in rects:
    #     autolabel(rect)

    fig.tight_layout()

    plt.savefig(results_file.replace('.pkl', '_{}.svg'.format(metric)))

backend = 'fake_melbourne'

layouts = ['chain', 'dense', 'noise_adaptive']

routings = ['basic', 'stochastic', 'lookahead']

metrics = ['hellinger', 'hog', 'ce', 'l1']

for layout in layouts:
    for routing in routings:
        for metric in metrics:
            plot('{}_results_{}_{}_{}.pkl'.format(backend, 'True', layout, routing), metric)
            plot('{}_results_{}_{}_{}.pkl'.format(backend, 'False', layout, routing), metric)


