#coding:utf-8
from sklearn.externals import joblib
from sklearn import metrics
import sys

if len(sys.argv)>=4:
    base_path=sys.argv[1]
    input_real_file=sys.argv[2]
    input_case_model=sys.argv[3]
else:
   # base_path = '/media/disk6/pathology_data_process/group19/'
    base_path = '/media/disk2/pathology_data_process_test28/group3/'
    input_real_file='/media/disk1/pathology_data_process/xiangya-real.txt'
   # input_real_file = '/media/disk4/pingkuang/pingkuang-real.txt'
    input_case_model='../../output/models/case_level-SVM-no'

input_feature_file=base_path+'heatmap_V3/case_level_feature'
output_file=base_path+'heatmap_V3/case_level_feature_real'
output_file2=base_path+'heatmap_V3/predict_real-by-feature'
output_statistics_file=base_path+'heatmap_V3/statistics-by-feature'     #statistics results

#base_path='/media/disk6/tcga/'
#input_real_file=base_path+'all-tcga-real.txt'
#input_feature_file=base_path+'all_case_level_feature'
#output_file=base_path+'all_case_level_feature_real'
#output_file2=base_path+'all_case_level_predict_real'

fw=open(output_file,'w')

real_lines=open(input_real_file).readlines()
predict_lines=open(input_feature_file).readlines()

for line in real_lines:
    id, label= line.strip().split()
    flag=0
    info=[]

    for line2 in reversed(predict_lines):
        may_id=line2.strip()[0:len(id)]

        if may_id!=id:
            continue
        else:
            parts = line2.strip().split()  # get features
            feature_5X = parts[-1].strip(',')
            feature_7X = parts[-2].strip(',')
            feature_10X = parts[-3].strip(',')

            info.append('%s %s  %s  %s    %s'%(id,feature_10X,feature_7X,feature_5X,label))

            predict_lines.remove(line2)
            flag=1

    if flag==1:
        max_norm=-1
        feature_5X='0'
        feature_7X='0'
        feature_10X='0'

        for one_info in info:
            parts = one_info.split()
            T_feature_5X = parts[-2]
            T_feature_7X = parts[-3]
            T_feature_10X = parts[-4]

            cur_norm = float(T_feature_5X)**2+float(T_feature_7X)**2+float(T_feature_10X)**2
            label=parts[-1]

            if cur_norm>max_norm:
                max_norm=cur_norm
                feature_5X = T_feature_5X
                feature_7X = T_feature_7X
                feature_10X = T_feature_10X

        one_info='%s %s  %s  %s    %s' % (id, feature_10X,feature_7X,feature_5X,label)

        fw.writelines(one_info + '\n')

 #   if flag==0:
  #      print('id '+id+' has no predict id')

#for 6 old pathlogy bit, it may be expand 7(add 0) or still be 6 bit
for line in real_lines:
    id, label= line.strip().split()
    flag=0
    info=[]

    if id[0]=='1':  #id with 1 begining is neglected, it is new id format
        continue

    #for old id format
    if  len(id)==6:
        id='0'+id     #expand 6 to 7 bit id, new id format
    elif len(id)==7 and id[0]=='0':
        id=id[1:]    #shrink 7 to 6 bit id, old id format because some files use old format

    for line2 in reversed(predict_lines):
        may_id=line2.strip()[0:len(id)]

        if may_id!=id:
            continue
        else:
            parts = line2.strip().split()  # get features
            feature_5X = parts[-1].strip(',')
            feature_7X = parts[-2].strip(',')
            feature_10X = parts[-3].strip(',')

            info.append('%s %s  %s  %s    %s' % (id, feature_10X, feature_7X, feature_5X, label))
            predict_lines.remove(line2)
            flag=1

    if flag==1:
        max_norm = -1
        feature_5X = '0'
        feature_7X = '0'
        feature_10X = '0'

        for one_info in info:
            parts = one_info.split()
            T_feature_5X = parts[-2]
            T_feature_7X = parts[-3]
            T_feature_10X = parts[-4]

            cur_norm = float(T_feature_5X) ** 2 + float(T_feature_7X) ** 2 + float(T_feature_10X) ** 2
            label = parts[-1]

            if cur_norm > max_norm:
                max_norm = cur_norm
                feature_5X = T_feature_5X
                feature_7X = T_feature_7X
                feature_10X = T_feature_10X

        one_info = '%s %s  %s  %s    %s' % (id, feature_10X, feature_7X, feature_5X, label)
        fw.writelines(one_info + '\n')

 #   if flag==0:
  #      print('id '+id+' has no predict id')


if len(predict_lines)!=0:
    #print('some predict has no real id')
    fw.writelines('\n'+'no label sample, and begin guess it is positive .....\n')

    for line in predict_lines:
        parts = line.strip().split()  # get features
        feature_5X = parts[-1].strip(',')
        feature_7X = parts[-2].strip(',')
        feature_10X = parts[-3].strip(',')
        id=parts[0]

        one_info='%s %s  %s  %s    %s' % (id, feature_10X, feature_7X, feature_5X, str(1))

        fw.writelines(one_info + '\n')

fw.close()

case_data=open(output_file).readlines()

fw=open(output_file2,'w')
clf=joblib.load(input_case_model)

result_info=[]

pos_count=0
neg_count=0
wrong_pos_count=0
wrong_neg_count=0

pos_y=[]
pos_y_scores=[]

for line in case_data:
    if len(line.strip())==0:
        continue

    if line[0:2]=='no':
        break

    id, feature_10X, feature_7X, feature_5X, label=line.strip().split()
    sample=[float(feature_10X),float(feature_7X),float(feature_5X)]

    if float(label) > 1:  #neglect wrong samples for image quality
        continue

    pre_label=clf.predict_proba([sample])

    info='%s %s  %s  %s    %s  [%s  %s]' % (id, feature_10X, feature_7X, feature_5X, label, str(pre_label[0][0]),str(pre_label[0][1]))

    if float(label) == 0:
        neg_count += 1
        pos_y_scores.append(float(pre_label[0][1]))
        pos_y.append(0)

    if float(label) == 1:
        pos_count += 1
        pos_y_scores.append(float(pre_label[0][1]))
        pos_y.append(1)

    if float(label) == 1 and float(pre_label[0][1]) <= 0.5:
        info=info+'          ?'
        wrong_pos_count+=1

    if float(label) ==0 and float(pre_label[0][0]) <= 0.5:
        info = info + '          ?'
        wrong_neg_count+=1

    result_info.append(info)
    fw.writelines(info + '\n')

fw.close()

fw=open(output_statistics_file,'w')
#comput accuracy and sensitivity and so on
accruray=float(pos_count+neg_count-wrong_pos_count-wrong_neg_count)/(pos_count+neg_count)
sensitivity=(pos_count-wrong_pos_count+1e-7)/(pos_count+1e-7)
specificity=(neg_count-wrong_neg_count+1e-7)/(neg_count+1e-7)
print('accuracy, senstivity, specificitty are:  %.4f, %.4f, %.4f'%(accruray,sensitivity,specificity))
fw.writelines('accuracy, sensitivity, specificity are:  %.4f, %.4f, %.4f\n'%(accruray,sensitivity,specificity))

print(pos_y)
print(pos_y_scores)

fw.writelines('real label: '+''.join([str(i)+' ' for i in pos_y])+'\n')
fw.writelines('predict value: '+''.join([str(i)+' ' for i in pos_y_scores])+'\n')

#compute AUC
if pos_count==0 or neg_count==0:
    exit()

test_auc = metrics.roc_auc_score(pos_y,pos_y_scores)
print('AUC is : %.4f'%(test_auc))

fw.writelines('AUC is: %.4f\n'%(test_auc))
fw.close()



