from param import parameter_parser
import load_data
from model import GCN
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
import evaluation_scores
import time
 


def CDA(n_fold):
    args = parameter_parser()
    dataset, cd_pairs = load_data.dataset(args)
 
    
    kf = KFold(n_splits = n_fold, shuffle = True)
    model = GCN(args)
    
    ave_acc = 0
    ave_prec = 0
    ave_sens = 0
    ave_f1_score = 0
    ave_mcc = 0
    ave_auc = 0
    ave_auprc = 0
    localtime = time.asctime( time.localtime(time.time()) )
    with open('./results/GraphCDA-100--5fold.txt', 'a') as f:
        f.write('time:\t'+ str(localtime)+"\n")

        
        for train_index, test_index in kf.split(cd_pairs):
            c_dmatix,train_cd_pairs,test_cd_pairs = load_data.C_Dmatix(cd_pairs,train_index,test_index)
            dataset['c_d']=c_dmatix
            score, cir_fea, dis_fea = load_data.feature_representation(model, args, dataset)
            train_dataset = load_data.new_dataset(cir_fea, dis_fea, train_cd_pairs)
            test_dataset = load_data.new_dataset(cir_fea, dis_fea, test_cd_pairs)
            X_train, y_train = train_dataset[:,:-2], train_dataset[:,-2:]
            X_test, y_test = test_dataset[:,:-2], test_dataset[:,-2:]
            print(X_train.shape,X_test.shape)
            clf = RandomForestClassifier(n_estimators=200,n_jobs=11,max_depth=20)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            y_pred = y_pred[:,0]
            y_prob = clf.predict_proba(X_test)
            y_prob = y_prob[1][:,0]
            tp, fp, tn, fn, acc, prec, sens, f1_score, MCC, AUC,AUPRC = evaluation_scores.calculate_performace(len(y_pred), y_pred, y_prob, y_test[:,0]) 
            print('RF: \n  Acc = \t', acc, '\n  prec = \t', prec, '\n  sens = \t', sens, '\n  f1_score = \t', f1_score, '\n  MCC = \t', MCC, '\n  AUC = \t', AUC,'\n  AUPRC = \t', AUPRC)
            f.write('RF: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(acc)+'\t  prec = \t'+ str(prec)+ '\t  sens = \t'+str(sens)+'\t  f1_score = \t'+str(f1_score)+ '\t  MCC = \t'+str(MCC)+'\t  AUC = \t'+ str(AUC)+'\t  AUPRC = \t'+ str(AUPRC)+'\n')
            ave_acc += acc
            ave_prec += prec
            ave_sens += sens
            ave_f1_score += f1_score
            ave_mcc += MCC
            ave_auc += AUC
            ave_auprc  += AUPRC
            
        ave_acc /= n_fold
        ave_prec /= n_fold
        ave_sens /= n_fold
        ave_f1_score /= n_fold
        ave_mcc /= n_fold
        ave_auc /= n_fold
        ave_auprc /= n_fold
        print('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n')
        f.write('Final: \t  tp = \t'+ str(tp) + '\t fp = \t'+ str(fp) + '\t tn = \t'+ str(tn)+ '\t fn = \t'+ str(fn)+'\t  Acc = \t'+ str(ave_acc)+'\t  prec = \t'+ str(ave_prec)+ '\t  sens = \t'+str(ave_sens)+'\t  f1_score = \t'+str(ave_f1_score)+ '\t  MCC = \t'+str(ave_mcc)+'\t  AUC = \t'+ str(ave_auc)+'\t  AUPRC = \t'+ str(ave_auprc)+'\n')
        

if __name__ == "__main__":
    
    n_fold = 5
    for i in range(100):
        CDA(n_fold)
