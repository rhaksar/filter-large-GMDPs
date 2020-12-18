import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42


def linear_approx():
    def log_linear(value, eps):
        if value < eps:
            value = eps
        return (np.log(eps)/(1-eps))*(1-value)

    eps_values = [1e-5, 1e-10, 1e-20]  # np.finfo(np.float).eps]
    style = ['-', '--', '-.']
    min_eps = min(eps_values)

    # x = np.linspace(min_eps, 1, num=1000, endpoint=True)
    x = np.logspace(-20, 0, num=1000, endpoint=True)

    log_values = [np.log(value) for value in x]

    fig = plt.figure(figsize=(10, 6))
    ax1 = plt.subplot(111)

    ax1.set_xlabel(r'input $\theta$', fontsize=16)
    ax1.set_ylabel(r'output f($\theta$)', fontsize=16)
    plt.tick_params(axis='both', which='major', labelsize=16, width=1.5)

    ax1.plot(x, log_values, linewidth=2, label='logarithm')
    for idx, eps in enumerate(eps_values):
        linear_values = [log_linear(value, eps) for value in x]
        ax1.plot(x, linear_values, linewidth=2, linestyle=style[idx], label='epsilon=%0.e' % eps)

    ax1.legend(fontsize=14)

    plt.show()
    # plt.savefig('log_approximation.pdf', dpi=300, bbox_inches='tight')


def non_uniform_parameter():
    dimension = 25

    alpha = dict()
    alpha_start = 0.1
    alpha_end = 0.4
    for r in range(dimension):
        for c in range(dimension):
            alpha[(r, c)] = alpha_start + (c / (dimension - 1)) * (alpha_end - alpha_start)

    alpha_dense = np.array([[alpha[(r, c)] for c in range(dimension)]
                            for r in range(dimension)])

    plt.imshow(alpha_dense, cmap='inferno', interpolation='nearest')
    plt.colorbar()
    plt.xlim([-0.5, dimension-0.5])
    plt.ylim([-0.5, dimension-0.5])
    plt.grid(which='major', axis='both', linewidth=1)
    plt.xticks(np.arange(-0.5, dimension-0.5, 1))
    plt.yticks(np.arange(-0.5, dimension-0.5, 1))
    plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False, bottom=False, left=False)

    plt.show()
    # plt.savefig('non_uniform_persistence_parameter.pdf', dpi=300, bbox_inches='tight')


def simulation_run():
    directory = 'data_fires_non_uniform/'
    filenames = ['lbp_d10_Kmax1_h3_s100.pkl', 'ravi_d10_Kmax1_eps1e-10_s100_new.pkl']
    seed = 1004  # seed = 1050

    obs_list = []
    flt_list = []
    for f in filenames:
        pkl_file = open(directory+f, 'rb')
        results = pickle.load(pkl_file)
        pkl_file.close()

        splits = f.split('_')
        method_name = splits[0]
        filter_accuracy = None
        if method_name == 'lbp':
            filter_accuracy = 'LBP_accuracy'

        elif method_name == 'ravi':
            filter_accuracy = 'RAVI_accuracy'

        obs = [100*e for e in results[seed]['observation_accuracy']]
        obs_list.append(obs)
        flt = [100*e for e in results[seed][filter_accuracy]]
        flt_list.append(flt)

    time = range(1, len(obs_list[0])+1)

    fig = plt.figure(figsize=(10, 6))
    plt.plot(time, obs_list[0], marker='o', linestyle=':', linewidth=2, color='gray', label='observation')
    plt.plot(time, flt_list[0], marker='o', linestyle='-', linewidth=2, color='C1', label='LBP, Kmax=1')
    plt.plot(time, flt_list[1], marker='o', color='C0', linewidth=2, label='RAVI, Kmax=1')

    plt.xlabel('Time', fontsize=18)
    plt.ylabel('Accuracy [%]', fontsize=18)
    plt.ylim(75, 102)
    # plt.xticks([1, 10, 20, 30, 40, 50])
    plt.tick_params(axis='both', which='major', labelsize=18, width=2)

    obs_median = np.median(obs_list[0])
    lbp_median = np.median(flt_list[0])
    ravi_median = np.median(flt_list[1])
    print(obs_median, lbp_median, ravi_median)

    plt.plot(time, obs_median*np.ones(len(obs_list[0])),
             linestyle=':', color='gray', label='observation median')
    plt.plot(time, lbp_median*np.ones(len(flt_list[0])),
             linestyle='--', color='C1', label='LBP, Kmax=1 median')
    plt.plot(time, ravi_median*np.ones(len(flt_list[1])),
             linestyle='--', color='C0', label='RAVI, Kmax=1 median')
    plt.legend(fontsize=14, ncol=2)

    # plt.show()
    plt.savefig('simulation_metric.pdf', dpi=300, bbox_inches='tight')


def timing_comparison():

    def autolabel(rects, string):
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x() + rect.get_width() / 2.0, 1.01*height,
                    '%s' % string, ha='center', va='bottom', fontsize=18)

    fig = plt.figure(figsize=(10, 6))
    # fig = plt.figure()

    ax = fig.add_subplot(111)
    ax.set_ylim(0, 107)
    ax.tick_params(axis='both', which='major', labelsize=18, width=2)
    ax.set_ylabel('Median Simulation Accuracy [%]', fontsize=18)
    ax.set_yticks(np.arange(0, 110, 10.0))

    ax2 = ax.twinx()
    ax2.set_ylim(10e-3, 10e1)
    # ax2.set_ylim(0.01, 14)
    ax2.set_yscale('log')
    ax2.tick_params(axis='both', which='major', labelsize=18, width=2)
    ax2.set_ylabel('Average time per update [seconds]', fontsize=18)

    width = 0.3
    spacing = 0

    directory = 'data_fires_non_uniform/'
    filenames = ['lbp_d10_Kmax1_h3_s100.pkl', 'lbp_d10_Kmax10_h3_s100.pkl',
                 'ravi_d10_Kmax1_eps1e-10_s100_new.pkl', 'ravi_d10_Kmax10_eps1e-10_s100_new.pkl']

    # directory = 'data_west_africa/'
    # filenames = ['lbp_wa_Kmax1_h3_s100.pkl', 'lbp_wa_Kmax10_h3_s100.pkl',
    #              'ravi_wa_Kmax1_eps1e-10_s100.pkl', 'ravi_wa_Kmax10_eps1e-10_s100.pkl']

    sim_acc = []
    timing = []
    for f in filenames:
        pkl_file = open(directory + f, 'rb')
        results = pickle.load(pkl_file)
        pkl_file.close()

        splits = f.split('_')
        method_name = splits[0]
        filter_accuracy = None
        if method_name == 'lbp':
            filter_accuracy = 'LBP_accuracy'
        elif method_name == 'ravi':
            filter_accuracy = 'RAVI_accuracy'

        obs_median_list = []
        filter_median_list = []
        update_average_list = []

        for e in results.keys():
            if isinstance(e, int):
                obs_median_list.append(np.median(results[e]['observation_accuracy']))
                filter_median_list.append(np.median(results[e][filter_accuracy]))
                update_average_list.append(np.mean(results[e]['time_per_update']))

        sim_acc.append(100*np.median(filter_median_list))
        timing.append(np.mean(update_average_list))

    print(sim_acc)
    print(timing)

    # median simulation accuracy and timing for LBP, Kmax=1
    rects = ax.bar(0, sim_acc[0], width, facecolor='None', edgecolor='black', linewidth=2)
    ax2.bar(0, timing[0], width, hatch='/', facecolor='None', edgecolor='black', linewidth=2)
    autolabel(rects, 'Kmax=1')

    # median simulation accuracy and timing for LBP, Kmax=10
    rects = ax.bar(0 + width + spacing, sim_acc[1], width, facecolor='None', edgecolor='black',
                   linewidth=2)
    ax2.bar(0 + width + spacing, timing[1], width, hatch='/', facecolor='None', edgecolor='black', linewidth=2)
    autolabel(rects, 'Kmax=10')

    # median simulation accuracy and timing for RAVI, Kmax=1
    rects = ax.bar(1, sim_acc[2], width, facecolor='None', edgecolor='black', linewidth=2)
    ax2.bar(1, timing[2], width, hatch='/', facecolor='None', edgecolor='black', linewidth=2)
    autolabel(rects, 'Kmax=1')

    # median simulation accuracy and timing for RAVI, Kmax=10
    rects = ax.bar(1 + width + spacing, sim_acc[3], width, facecolor='None', edgecolor='black',
                   linewidth=2)
    ax2.bar(1 + width + spacing, timing[3], width, hatch='/', facecolor='None', edgecolor='black', linewidth=2)
    autolabel(rects, 'Kmax=10')

    ax.set_xticks(np.arange(2) + width / 2)
    ax.set_xticklabels(('LBP', 'RAVI'), fontsize=18)

    plt.show()
    # plt.savefig('ff_10x10_timing.pdf', dpi=300, bbox_inches='tight')


def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color=color)
    plt.setp(bp['medians'], color='gray')
    plt.setp(bp['means'], color='gray')


def boxplots():

    results_set = 'west_africa'
    if results_set == 'west_africa':
        folder = 'data_west_africa/'
        files = ['lbp_wa_Kmax1_h3_s100.pkl', 'lbp_wa_Kmax10_h3_s100.pkl',
                 'ravi_wa_Kmax1_eps1e-10_s100.pkl', 'ravi_wa_Kmax10_eps1e-10_s100.pkl']

    obs_median_list = []
    filter_median_list = []
    for f in files:
        pkl_file = open(folder + f, 'rb')
        results = pickle.load(pkl_file)
        pkl_file.close()

        splits = f.split('_')
        method_name = splits[0]
        filter_accuracy = None
        if method_name == 'lbp':
            filter_accuracy = 'LBP_accuracy'
        elif method_name == 'ravi':
            filter_accuracy = 'RAVI_accuracy'

        obs_median_list.append([np.median(results[e]['observation_accuracy']) for e in results.keys()
                                if isinstance(e, int)])
        filter_median_list.append([np.median(results[e][filter_accuracy]) for e in results.keys()
                                   if isinstance(e, int)])

    plt.boxplot(filter_median_list, whis='range')
    plt.show()


def median_graph():
    directory = ''
    filename = 'ravi_d10_Kmax10_eps1e-10_s100.pkl'
    pkl_file = open(directory + filename, 'rb')
    results = pickle.load(pkl_file)
    pkl_file.close()

    splits = filename.split('_')
    method_name = splits[0]
    filter_accuracy = None
    if method_name == 'lbp':
        print('method: LBP')
        print('horizon:', results['horizon'])
        filter_accuracy = 'LBP_accuracy'

    elif method_name == 'ravi':
        print('method: RAVI')
        print('epsilon:', results['epsilon'])
        filter_accuracy = 'RAVI_accuracy'

    if 'dimension' in results.keys():
        print('dimension:', results['dimension'])
    print('Kmax:', results['Kmax'])

    seed_list = []
    obs_median_list = []
    filter_median_list = []
    update_average_list = []
    for e in results.keys():
        if isinstance(e, int):
            obs_median_list.append(np.median(results[e]['observation_accuracy']))
            filter_median_list.append(np.median(results[e][filter_accuracy]))
            update_average_list.append(np.mean(results[e]['time_per_update']))
            seed_list.append(e)

    plt.plot(seed_list, filter_median_list, '.')
    plt.show()


if __name__ == '__main__':
    # linear_approx()
    # non_uniform_parameter()
    # simulation_run()
    # timing_comparison()
    # boxplots()
    # median_graph()
    pass
