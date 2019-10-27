import matplotlib.pyplot as plt
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')


def leafcounter(node):
    '''
    Description: helper function to count the number of lists in a dicitonary.
                 Knowing the number of data to plot will determine the number
                 of distinct color to create
    Params: node = the dictionary containing saved data
    Outputs: the number of lists to plot
    '''
    if isinstance(node, dict):
        return sum([leafcounter(node[n]) for n in node])
    else:
        return 1


def get_spaced_colors(n):
    '''
    Description: generate n equally spaced colors
    Params: n = the number of color to generate
    Outputs: a generator containing n generated colors
    '''
    cm = plt.get_cmap('gist_rainbow')
    for i in range(n):
        yield cm(1. * i / n)


def plot_list_variables(data, dataset_name, x_axis_name, y_axis_name, save_dir):
    '''
    Description: plot non nested dictionary of data
    Params: data = the dictionary to plot
            x_axis_name = the name of the x-axis
            y_axis_name = the name of the y-axis
            save_dir = the directory where to save the figures
    Outputs: Pre-processed image features and labels
    '''
    colors = get_spaced_colors(len(data.keys()))
    for algo_name, algo_data in data.items():
        plt.plot(list(np.arange(len(algo_data))), algo_data, color=next(colors))
    plt.legend(tuple(list(data.keys())))

    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.ylim((0, 100))
    plt.title('{} on {}'.format(y_axis_name.replace("_", " "), dataset_name))
    plt.savefig('{}/{}.png'.format(save_dir, y_axis_name.replace(" ", "_")))
    plt.clf()


def plot_dict_variables(data, dataset_name, x_axis_name, y_axis_name, save_dir):
    '''
    Description: plot nested dictionary of data
    Params: data = the dictionary to plot
            x_axis_name = the name of the x-axis
            y_axis_name = the name of the y-axis
            save_dir = the directory where to save the figures
    Outputs: Pre-processed image features and labels
    '''
    colors = get_spaced_colors(leafcounter(data))
    labels = []
    for algo_name, algo_data in data.items():
        for i, (layer, layer_data) in enumerate(algo_data.items()):
            plt.plot(list(np.arange(len(layer_data))), layer_data, color=next(colors))
            labels.append("{} {}".format(algo_name, layer))
    plt.legend(labels)

    plt.xlabel(x_axis_name)
    plt.ylabel(y_axis_name)
    plt.ylim((0, 120))
    plt.title('{} on {}'.format(y_axis_name.replace("_", " "), dataset_name))
    plt.savefig('{}/{}.png'.format(save_dir, y_axis_name.replace(" ", "_")))
    plt.clf()


def read_data():
    '''
    Description: read the json files of all experiments
    Outputs: a dictionary containing all the json files
    '''
    json_files = {}
    for subdir, dirs, files in os.walk('./experiments'):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(subdir, file), 'r') as f:
                    algo = subdir.split("/")[-1].split('-')[0]
                    dataset = subdir.split("/")[-2]
                    if not dataset in json_files.keys():
                        json_files[dataset] = {}
                    json_files[dataset][algo] = json.load(f)
    return json_files


def generate_dataset_figures(json_files, dataset_name, save_dir):
    '''
    Description: generate the figures of all experiments
    Params: data = the data dictionary to all experiments
            save_dir = the directory where to save the figures
    Outputs: Pre-processed image features and labels
    '''
    # Initialize variables to plot
    losses = {algo: [] for algo in json_files.keys()}
    test_accuracies = {algo: [] for algo in json_files.keys()}
    delta_angles = {algo: {} for algo in json_files.keys() if algo != 'bp'}
    weight_angles = {algo: {} for algo in json_files.keys() if algo != 'bp'}

    for algo_name, algo_data in json_files.items():

        for _, epoch_data in algo_data.items():
            losses[algo_name].append(epoch_data['loss'])
            test_accuracies[algo_name].append(epoch_data['test_accuracy'])
            if algo_name != 'bp':
                for layer, angle in epoch_data['delta_angles'].items():
                    if layer in delta_angles[algo_name].keys():
                        delta_angles[algo_name][layer].append(angle)
                    else:
                        delta_angles[algo_name][layer] = [angle]

                for layer, angle in epoch_data['weight_angles'].items():
                    if layer in weight_angles[algo_name].keys():
                        weight_angles[algo_name][layer].append(angle)
                    else:
                        weight_angles[algo_name][layer] = [angle]

    # plot the `loss`, `test_accuracy`, `delta_angles` and `weight_angles`
    plot_list_variables(losses, dataset_name, 'epochs', 'loss', save_dir)
    plot_list_variables(test_accuracies, dataset_name, 'epochs', 'test_accuracies', save_dir)
    plot_dict_variables(delta_angles, dataset_name, 'epochs', 'delta_angles', save_dir)
    plot_dict_variables(weight_angles, dataset_name, 'epochs', 'weight_angles', save_dir)


if __name__ == '__main__':
    json_files = read_data()

    # Generate figures
    for dataset_name, data in json_files.items():
        save_dir = os.path.join('figures', dataset_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        generate_dataset_figures(data, dataset_name, save_dir)
