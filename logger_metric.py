import numpy as np

def get_errors_helper(error_list, class_list, class_enum):
    errors = {} # list per class
    for c in class_enum:
        errors[str(c)] = []
    for e,c in zip(error_list, class_list):
        errors[str(class_enum(c))].append(e)
    per_class_stats = {}
    ang30s = []
    ang15s = []
    ang7_5s = []
    medians_list = []
    for classname, values in errors.items():
        if len(values) > 0:
            values = np.array(values)
            ang30 = np.mean(values < 30)
            ang15 = np.mean(values < 15)
            ang7_5 = np.mean(values < 7.5)
            median = np.median(values)
            medians_list.append(median)
            ang30s.append(ang30)
            ang15s.append(ang15)
            ang7_5s.append(ang7_5)
            per_class_stats[classname] = (median, ang30, ang15, ang7_5, values)
    return [np.mean(medians_list), np.mean(ang30s), np.mean(ang15s), np.mean(ang7_5s), error_list], per_class_stats

def get_errors(error_list, class_list, hard_list, class_enum):
    easy_error_list = []
    easy_class_list = []
    for e,c,h in zip(error_list, class_list, hard_list):
        if not h:
            easy_error_list.append(e)
            easy_class_list.append(c)
    easy_stats = get_errors_helper(easy_error_list, easy_class_list, class_enum)
    stats = get_errors_helper(error_list, class_list, class_enum)
    return easy_stats, stats

def print_stats(stats, verbose=False):
    print('mean over median: {}'. format(stats[0][0]))
    print('angle 30: {}'. format(stats[0][1]))
    if verbose:
        print('angle 15: {}'. format(stats[0][2]))
        print('angle 7.5: {}'. format(stats[0][3]))
        for name, median in stats[1].items():
            print('{} median: {}'.format(name, median))
