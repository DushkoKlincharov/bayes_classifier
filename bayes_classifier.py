import pprint as pp
import numpy as np
from scipy.stats import norm

# [diplomiral, uspeh, vid, prosek, nepolozeni]

train_data = [[0,  4.5, 0, 8.31, 1],
              [0,    5, 0,   10, 0],
              [0,    5, 1, 7.98, 0],
              [0, 4.98, 0, 9.42, 0],
              [0,    4, 0, 8.98, 1],
              [1, 4.23, 0, 8.23, 2],
              [1,    3, 1, 8.56, 1],
              [1,    5, 0,   10, 2],
              [1, 4.87, 0, 7.99, 3],
              [1, 4.56, 1, 8.98, 2],
              [2, 3.92, 1,    7, 2],
              [2, 3.45, 1,    5, 3],
              [2, 2.67, 1,  6.3, 2],
              [2,  3.2, 1,  8.2, 3],
              [2,    2, 0,  6.2, 3]]

meta_data = [('discrete', [0, 1, 2]),
             ('continuous'),
             ('discrete', [0, 1]),
             ('continuous'),
             ('discrete', [0, 1, 2, 3])]

def discrete_prob_dist(category, parameter):
    need_for_laplace = False
    distribution = {x: 0 for x in meta_data[parameter][1]}
    category_data = [x for x in train_data if x[0] == category]
    for sample in category_data:
        distribution[sample[parameter]] += 1
    for x in distribution.keys():
        distribution[x] = distribution[x] * 1.0 / len(category_data)
        if distribution[x] == 0:
            need_for_laplace = True
    if(need_for_laplace):
        p = 1.0 / len(meta_data[parameter][1])
        for x in distribution.keys():
            distribution[x] = round((distribution[x] * len(category_data) + p) * 1.0 / (len(category_data) + 1), 3)
    return distribution

def continuous_prob_dist(category, parameter):
    param_data = [x[parameter] for x in train_data if x[0] == category]
    return {'mean': round(np.mean(param_data),3), 'std': round(np.std(param_data),3)}

classifier = {}
for i in range(len(meta_data[0][1])):
    category = meta_data[0][1][i]
    classifier[category] = {'p': sum([1 for x in train_data if x[0] == category]) * 1.0 / len(train_data)}
    for j in range(1, len(meta_data)):
        type_of_param = meta_data[j][0]
        if type_of_param == 'discrete':
            classifier[i][j] = discrete_prob_dist(category, j)
        else:
            classifier[i][j] = continuous_prob_dist(category, j)

pp.pprint(classifier, width=2)

test_data = [[4.7, 0, 9, 1],
             [  3, 0, 9, 1],
             [  3, 1, 8, 3]]

for sample in test_data:
    print
    prob = {0: 1, 1: 1, 2: 1}
    for cat in meta_data[0][1]:
        prob[cat] *= classifier[cat]['p']
        for param in range(1, len(meta_data)):
            if meta_data[param][0] == 'discrete':
                prob[cat] *= classifier[cat][param][sample[param-1]]
            else:
                prob[cat] *= norm(classifier[cat][param]['mean'], classifier[cat][param]['std']).pdf(sample[param-1])
    s = sum(prob.values())
    for key in prob:
        print('{}: {:.3f}%'.format(key, float(prob[key]/s)*100))