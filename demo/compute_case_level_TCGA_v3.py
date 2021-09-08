from sklearn import cross_validation,metrics

base_path='/media/disk6/tcga/'
input_predict_file=base_path+'all_case_level_predict_real'
input_real_file=base_path+'all-tcga-real.txt'

real_lines=open(input_real_file).readlines()
predict_lines=open(input_predict_file).readlines()
case_ids={}
all_ids=[]

for line in real_lines:
    id, label=line.strip().split()
    id=id[0:12]     #case level id

    all_ids.append(id)

    #positive case if one slide is positive
    if not case_ids.has_key(id):
        case_ids[id]=int(label)
    else:
        if case_ids[id]<int(label):
            case_ids[id] = int(label)

all_ids=list(set(all_ids))
pos_y=[]
pos_y_scores=[]
for one_id in all_ids:
    label=case_ids[one_id]

    pos_y.append(label)

    temp_score=0
    for line in predict_lines:
        parts=line.strip().split()
        parts=parts[-1].strip(']').split()
        pre_score=float(parts[-1])

        if label>0:   #positive
            if pre_score>temp_score:
                temp_score=pre_score
        else:    #neg
            if pre_score<temp_score:
                temp_score=pre_score

    pos_y_scores.append(temp_score)

test_auc = metrics.roc_auc_score(pos_y,pos_y_scores)
print('AUC is : %.4f'%(test_auc))

print(pos_y)
print(pos_y_scores)





