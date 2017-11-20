import pprint as pp
import numpy as np
from scipy.stats import norm

# [diplomiral, uspeh, vid, prosek, nepolozeni]

meta_data = [('discrete', [0, 1, 2]),  # na vreme -> 0, so zadocnuvanje -> 1, ne diplomiral -> 2
             ('continuous', float('inf')),  # uspeh sredno 
             ('discrete', [0, 1]),  # gimnazisko -> 0, strucno -> 1
             ('continuous', float('inf')),  # prosek prva
             ('discrete', [0, 1, 2, 3])]  # 0 -> 0, 1-2 -> 1, 3-5 -> 2, >5 -> 3

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

def discrete_prob_dist(category, parameter):
    """Vraka dictionary {x1:p1, x2:p2, ..., xn: pn}
        x1, x2, ..., xn se vrednosti koi moze da gi dobie parametarot
        niv gi gledame od meta_data[parameter][1]"""
    need_for_laplace = False
    distribution = {x: 0 for x in meta_data[parameter][1]}  # inicijalizacija {x1:0, x2:0, ..., xn:0}
    category_data = [x for x in train_data if x[0] == category]  # gi izdvojuvame primerocite od odredenata kategorija
    for sample in category_data:
        distribution[sample[parameter]] += 1  #  broime po kolku pati se pojavuva sekoja vrednost na parametarot
    for x in distribution.keys():
        distribution[x] = distribution[x] * 1.0 / len(category_data)  #  delime so vkupen broj na primeroci za da dobieme verojatnost
        if distribution[x] == 0:
            need_for_laplace = True
    if(need_for_laplace):
        p = 1.0 / len(meta_data[parameter][1])  # na sekoja vrednost na parametarot dodavame uniformna raspredelba
        for x in distribution.keys():
            distribution[x] = round((distribution[x] * len(category_data) + p) * 1.0 / (len(category_data) + 1), 3)
    return distribution

def continuous_prob_dist(category, parameter):
    """Vraka dictionary {'mean': mi, 'std': sigma}"""
    param_data = [x[parameter] for x in train_data if x[0] == category]  # lista od vrednostite od parametarot na odredenata kategorija
    return {'mean': round(np.mean(param_data),3), 'std': round(np.std(param_data),3)}

# implementacija na modelot
# classifier e dictinary od oblik
# cat_1:{
#           'p': p(cat_1),
#           param_1: {prob_dist},
#           param_2: {prob_dist},
#           ...
#           param_n: {prob_dist}
#}
# cat_2:{
#           'p': p(cat_2),
#           param_1: {prob_dist},
#           param_2: {prob_dist},
#           ...
#           param_n: {prob_dist}  
# }
# ...
# cat_n:{
#           'p': p(cat_n),
#           param_1: {prob_dist},
#           param_2: {prob_dist},
#           ...
#           param_n: {prob_dist}  
# }
classifier = {}
for i in range(len(meta_data[0][1])):  # vo prviot element na meta_data se naoga koi se kategoriite
    category = meta_data[0][1][i]
    classifier[category] = {'p': round(sum([1 for x in train_data if x[0] == category]) * 1.0 / len(train_data), 3)}  # odreduvanje na p(c1), p(c2), ..., p(cn)
    for j in range(1, len(meta_data)):  # narednite elementi na meta_data se parametrite (karakteristiki), i za sekoja posebno odreduvame distribucija
        type_of_param = meta_data[j][0]
        if type_of_param == 'discrete':
            classifier[category][j] = discrete_prob_dist(category, j)
        else:
            classifier[category][j] = continuous_prob_dist(category, j)

pp.pprint(classifier, width=2)

test_data = [[4.7, 0, 9, 1],
             [  3, 0, 9, 1],
             [  3, 1, 8, 3]]

for sample in test_data:
    print('\n{}'.format(sample))
    prob = {}
    for x in meta_data[0][1]:   # inicijalizacija na verojatnostite primerokot da se naoga vo i-tata klasa
        prob[x] = 1
    for cat in meta_data[0][1]:
        prob[cat] *= classifier[cat]['p']
        for param in range(1, len(meta_data)):
            if meta_data[param][0] == 'discrete':
                prob[cat] *= classifier[cat][param][sample[param-1]]  # za diskretni parametri mnozime po frekfencijata koja ja ima vrednosta za toj parametar i taa klasa od classifier
            else:
                prob[cat] *= norm(classifier[cat][param]['mean'], classifier[cat][param]['std']).pdf(sample[param-1])  # za neprekinati parametri mnozime po gustinata na gausovata raspredelba so parametri za toj parametar od taa kategorija 
    s = sum(prob.values())
    for key in prob:
        print('{}: {:.3f}%'.format(key, float(prob[key]/s)*100))