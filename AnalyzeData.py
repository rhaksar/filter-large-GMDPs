import numpy as np
import pickle

if __name__ == '__main__':
    # directory = 'data_fires_uniform/'
    # filename = 'ravi_d10_Kmax10_eps1e-05_s100.pkl'
    # filename = 'lbp_d3_Kmax10_h3_s100.pkl'

    directory = 'data_fires_non_uniform/'
    # filename = 'ravi_d25_Kmax10_eps1e-05_s100.pkl'
    # filename = 'lbp_d10_Kmax1_h3_s100.pkl'
    filename = 'ravi_d25_Kmax10_eps1e-10_s100_new.pkl'

    # directory = 'data_west_africa/'
    # filename = 'ravi_wa_Kmax10_eps1e-10_s100_new.pkl'
    # filename = 'lbp_wa_Kmax10_h3_s100.pkl'

    pkl_file = open(directory+filename, 'rb')
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

    obs_median_list = []
    filter_median_list = []
    update_average_list = []
    for e in results.keys():
        if isinstance(e, int):
            obs_median_list.append(np.median(results[e]['observation_accuracy']))
            filter_median_list.append(np.median(results[e][filter_accuracy]))
            update_average_list.append(np.mean(results[e]['time_per_update']))

    print('observation/filter min median accuracy: %0.2f / %0.2f'
          % (np.amin(obs_median_list)*100, np.amin(filter_median_list)*100))
    print('observation/filter median median accuracy: %0.2f / %0.2f'
          % (np.median(obs_median_list)*100, np.median(filter_median_list)*100))
    print('observation/filter max median accuracy: %0.2f / %0.2f'
          % (np.amax(obs_median_list)*100, np.amax(filter_median_list)*100))
    print('required %0.6fs per update on average' % np.mean(update_average_list))
