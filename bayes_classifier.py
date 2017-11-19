import pprint as pp
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

meta_data = [('descrete', [0, 1, 2]),
             ('continuous'),
             ('descrete', [0, 1]),
             ('continuous'),
             ('descrete', [0, 1, 2, 3])]

def descrete_prob_dist(category, parameter):
    
    for sample in [x for x in train_data if x[0] == category]:
        

classifier = {}
for i in range(len(meta_data[0][1])):
    category = meta_data[0][1][i]
    classifier[category] = {}
    for j in range(1, len(meta_data)):
        type_of_param = meta_data[j][0]
        if type_of_param == 'descrete':
            classifier[i][j] = {x: 0 for x in meta_data[j][1]}
        else:
            classifier[i][j] = {'mean': 0, 'std': 0}

#pp.pprint(classifier, width=1)

