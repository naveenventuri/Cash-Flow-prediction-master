import numpy as np
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import pandas as pd
import Dataset as ds
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

def elbow_fit(x_train, plot_graph=False):
    max_features = 2000000
    maxlen = 100
    batch_size = 32
    
    # k means determine k
    distortions = []
    K = range(1,100)
    for k in K:
        model = KMeans(n_clusters=k).fit(x_train)
        
        distortions.append(sum(np.min(cdist(x_train, model.cluster_centers_, 'euclidean'), axis=1)) / len(x_train))
        
        print("KMeans computation for cluster count, distance : ", k, sum(np.min(cdist(x_train, model.cluster_centers_, 'euclidean'), axis=1)) / len(x_train))
    
    if plot_graph:
        # Plot the elbow
        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        plt.savefig("test.png")
        plt.show()

def tc_420k():
    (x_train, x_test, y_train, y_test) = ds.load(ds.ALL_FILE_704, test_size=0.0, exclude_cols=['ZIP_CODE', 'ZIP_CODE_5', 'bet_30', 'bet_60','bet_90','bet_120','bet_150','bet_180','AI_HasBBRecord','AI_HasCreditHistory', 'CAPITAL_DEFICIT_ROOT', 'CAPITAL_SURPLUS_ROOT', 'EST_INVOICE_TOT_6M_SQR', 'EST_INVOCIE_TOT_6M_ROOT', 'PUBLIC_REMARK_24M_SQR', 'PUBLIC_REMARK_24M_ROOT' , 'PUBLIC_Remark_12M_ROOT'], y_cols=['PersonID'])
    
    elbow_fit(x_train)

def tc_save_420k():
    (x_train, x_test, y_train, y_test) = ds.load(ds.ALL_FILE_704 , test_size=0.0, exclude_cols=['ZIP_CODE', 'ZIP_CODE_5', 'bet_30', 'bet_60','bet_90','bet_120','bet_150','bet_180','AI_HasBBRecord','AI_HasCreditHistory', 'CAPITAL_DEFICIT_ROOT', 'CAPITAL_SURPLUS_ROOT', 'EST_INVOICE_TOT_6M_SQR', 'EST_INVOCIE_TOT_6M_ROOT', 'PUBLIC_REMARK_24M_SQR', 'PUBLIC_REMARK_24M_ROOT' , 'PUBLIC_Remark_12M_ROOT'], y_cols=['PersonID'])

    model = KMeans(n_clusters=30).fit(x_train)

    personIds = np.array(y_train)
    clusters = model.labels_
    personIds = personIds.reshape(personIds.shape[0],1).ravel()
    clusters = clusters.reshape(clusters.shape[0],1).ravel()
    
    d = {'PersonID': personIds, 'CLUSTER': clusters}
    df = pd.DataFrame(data=d)
        
    df.to_csv(config['DEFAULT']['PERSON_CLUSTERS'])
    
def tc_elbow_zip():
    (x_train, x_test, y_train, y_test) = ds.load(ds.ALL_FILE_704, test_size=0.0, exclude_cols=['IS_CURR_CUST','EST_INV_TOT','EST_INVOICE_FIXED','ACT_INVOICE_FIXED','SUBS_FIXED_CNT','SUBS_FIXED_APP_CNT','EST_INVOICE_MOB','ACT_INVOICE_MOB','SUBS_MOB_CNT','SUBS_MOB_APP_CNT','EST_INVOICE_INTERNET','ACT_INVOICE_INTERNET','SUBS_INTERNET_CNT','SUBS_INTENET_APP_CNT','EST_INVOICE_OTHER','ACT_INVOICE_OTHER','SUBS_OTHER_CNT','SUBS_OTHER_APP_CNT','INVOICE_TOT_6M','PUBLIC_REMARK_DAYS','PRIVATE_REMARK_DAYS','EST_INVOICE_INCOME','DEBT_COLL_CASES_12M','DEBT_COLL_CASES_24M','PRIVATE_REMARKS_24M','PUBLIC_REMARK_24M','PUBLIC_REMARK_24M_SQR','PUBLIC_REMARK_24M_ROOT','PUBLIC_Remark_12M_ROOT','MAX_STATUS_12M','AGE','GENDER','PAYORDERS_CNT_12M','DISTRESS_CNT_24M','LATE_FEE_TAX_RETURN','CAPITAL_DEFICIT','PROPERTY_TAX','IS_NO_MISSING_INCOME','CAPITAL_DEFICIT_ROOT','CAPITAL_SURPLUS_ROOT','PRIVATE_DEBT_EA','EST_INVOICE_TOT_6M_SQR','EST_INVOCIE_TOT_6M_ROOT','CAPITAL_DEFICIT_PREV_YEAR','DEBT_EA','PAYMENT_REMARKS_36M','APPLIED_AMOUNT','IS_DC_ALL','IS_DC_FC','IS_DC_TF','DC_CNT','PUBLIC_DEBT_CNT','PRIVATE_DEBT_CNT','CLAIMED_CAPITAL_TOT','PUBLIC_DEBT_AMT','PRIVATE_DEBT_AMT','CLAIMED_CAPITAL_PAID','CLAIMED_CAPITAL_FC','CLAIMED_CAPITAL_TF','IS_MISSING_ADDRESS','INCOME_PREV_12M','INCOME_PREV_24M','INCOME_PREV_36M','LOSS_ON_CAPITAL_PREV_12M','LOSS_ON_CAPITAL_PREV_24M','LOSS_ON_CAPITAL_PREV_36M','ADDITIONAL_TAX_PREV_12M','ADDITIONAL_TAX_PREV_24M','ADDITIONAL_TAX_PREV_36M','LATE_FEE_PREV_12M','LATE_FEE_PREV_24M','LATE_FEE_PREV_36M','TYPE_ZERO_TAX_PREV_12M','TYPE_ZERO_TAX_PREV_24M','TYPE_ZERO_TAX_PREV_36M','ZIP_CODE_5','DECEASED','AI_IsBB','CreditLimitApproved','AccumulatedSales','TotalDebt','TotalDebtCollection','RemainingAmountCollection','bet_30', 'bet_60','bet_90','bet_120','bet_150','bet_180','AI_HasBBRecord','AI_HasCreditHistory'], y_cols=['PersonID'])
    elbow_fit(x_train)

def tc_elbow_seg_score():
    (x_train, x_test, y_train, y_test) = ds.load(ds.ALL_FILE_704, test_size=0.0, exclude_cols=['ZIP_CODE', 'IS_CURR_CUST','EST_INV_TOT','EST_INVOICE_FIXED','ACT_INVOICE_FIXED','SUBS_FIXED_CNT','SUBS_FIXED_APP_CNT','EST_INVOICE_MOB','ACT_INVOICE_MOB','SUBS_MOB_CNT','SUBS_MOB_APP_CNT','EST_INVOICE_INTERNET','ACT_INVOICE_INTERNET','SUBS_INTERNET_CNT','SUBS_INTENET_APP_CNT','EST_INVOICE_OTHER','ACT_INVOICE_OTHER','SUBS_OTHER_CNT','SUBS_OTHER_APP_CNT','INVOICE_TOT_6M','PUBLIC_REMARK_DAYS','PRIVATE_REMARK_DAYS','EST_INVOICE_INCOME','DEBT_COLL_CASES_12M','DEBT_COLL_CASES_24M','PRIVATE_REMARKS_24M','PUBLIC_REMARK_24M','PUBLIC_REMARK_24M_SQR','PUBLIC_REMARK_24M_ROOT','PUBLIC_Remark_12M_ROOT','MAX_STATUS_12M','AGE','GENDER','PAYORDERS_CNT_12M','DISTRESS_CNT_24M','LATE_FEE_TAX_RETURN','CAPITAL_DEFICIT','PROPERTY_TAX','IS_NO_MISSING_INCOME','CAPITAL_DEFICIT_ROOT','CAPITAL_SURPLUS_ROOT','PRIVATE_DEBT_EA','EST_INVOICE_TOT_6M_SQR','EST_INVOCIE_TOT_6M_ROOT','CAPITAL_DEFICIT_PREV_YEAR','DEBT_EA','PAYMENT_REMARKS_36M','APPLIED_AMOUNT','IS_DC_ALL','IS_DC_FC','IS_DC_TF','DC_CNT','PUBLIC_DEBT_CNT','PRIVATE_DEBT_CNT','CLAIMED_CAPITAL_TOT','PUBLIC_DEBT_AMT','PRIVATE_DEBT_AMT','CLAIMED_CAPITAL_PAID','CLAIMED_CAPITAL_FC','CLAIMED_CAPITAL_TF','IS_MISSING_ADDRESS','INCOME_PREV_12M','INCOME_PREV_24M','INCOME_PREV_36M','LOSS_ON_CAPITAL_PREV_12M','LOSS_ON_CAPITAL_PREV_24M','LOSS_ON_CAPITAL_PREV_36M','ADDITIONAL_TAX_PREV_12M','ADDITIONAL_TAX_PREV_24M','ADDITIONAL_TAX_PREV_36M','LATE_FEE_PREV_12M','LATE_FEE_PREV_24M','LATE_FEE_PREV_36M','TYPE_ZERO_TAX_PREV_12M','TYPE_ZERO_TAX_PREV_24M','TYPE_ZERO_TAX_PREV_36M','ZIP_CODE_5','DECEASED','AI_IsBB','CreditLimitApproved','AccumulatedSales','TotalDebt','TotalDebtCollection','RemainingAmountCollection','bet_30', 'bet_60','bet_90','bet_120','bet_150','bet_180','AI_HasBBRecord','AI_HasCreditHistory'], y_cols=['PersonID'])
    elbow_fit(x_train)
    
def tc_save_zip_clusters():
    (x_train, x_test, y_train, y_test) = ds.load(ds.ALL_FILE_704, test_size=0.0, exclude_cols=['IS_CURR_CUST','EST_INV_TOT','EST_INVOICE_FIXED','ACT_INVOICE_FIXED','SUBS_FIXED_CNT','SUBS_FIXED_APP_CNT','EST_INVOICE_MOB','ACT_INVOICE_MOB','SUBS_MOB_CNT','SUBS_MOB_APP_CNT','EST_INVOICE_INTERNET','ACT_INVOICE_INTERNET','SUBS_INTERNET_CNT','SUBS_INTENET_APP_CNT','EST_INVOICE_OTHER','ACT_INVOICE_OTHER','SUBS_OTHER_CNT','SUBS_OTHER_APP_CNT','INVOICE_TOT_6M','PUBLIC_REMARK_DAYS','PRIVATE_REMARK_DAYS','EST_INVOICE_INCOME','DEBT_COLL_CASES_12M','DEBT_COLL_CASES_24M','PRIVATE_REMARKS_24M','PUBLIC_REMARK_24M','PUBLIC_REMARK_24M_SQR','PUBLIC_REMARK_24M_ROOT','PUBLIC_Remark_12M_ROOT','MAX_STATUS_12M','AGE','GENDER','PAYORDERS_CNT_12M','DISTRESS_CNT_24M','LATE_FEE_TAX_RETURN','CAPITAL_DEFICIT','PROPERTY_TAX','IS_NO_MISSING_INCOME','CAPITAL_DEFICIT_ROOT','CAPITAL_SURPLUS_ROOT','PRIVATE_DEBT_EA','EST_INVOICE_TOT_6M_SQR','EST_INVOCIE_TOT_6M_ROOT','CAPITAL_DEFICIT_PREV_YEAR','DEBT_EA','PAYMENT_REMARKS_36M','APPLIED_AMOUNT','IS_DC_ALL','IS_DC_FC','IS_DC_TF','DC_CNT','PUBLIC_DEBT_CNT','PRIVATE_DEBT_CNT','CLAIMED_CAPITAL_TOT','PUBLIC_DEBT_AMT','PRIVATE_DEBT_AMT','CLAIMED_CAPITAL_PAID','CLAIMED_CAPITAL_FC','CLAIMED_CAPITAL_TF','IS_MISSING_ADDRESS','INCOME_PREV_12M','INCOME_PREV_24M','INCOME_PREV_36M','LOSS_ON_CAPITAL_PREV_12M','LOSS_ON_CAPITAL_PREV_24M','LOSS_ON_CAPITAL_PREV_36M','ADDITIONAL_TAX_PREV_12M','ADDITIONAL_TAX_PREV_24M','ADDITIONAL_TAX_PREV_36M','LATE_FEE_PREV_12M','LATE_FEE_PREV_24M','LATE_FEE_PREV_36M','TYPE_ZERO_TAX_PREV_12M','TYPE_ZERO_TAX_PREV_24M','TYPE_ZERO_TAX_PREV_36M','ZIP_CODE_5','DECEASED','AI_IsBB','CreditLimitApproved','AccumulatedSales','TotalDebt','TotalDebtCollection','RemainingAmountCollection','bet_30', 'bet_60','bet_90','bet_120','bet_150','bet_180','AI_HasBBRecord','AI_HasCreditHistory'], y_cols=['PersonID'])
    
    model = KMeans(n_clusters=20).fit(x_train)
    
    zips = np.array(x_train['ZIP_CODE'])
    clusters = model.labels_
     
    d = {'ZIP_CODE': zips, 'ZIP_CLUSTER': clusters}
    df = pd.DataFrame(data=d)
    
    df.to_csv(config['DEFAULT']['ZIP_CLUSTERS'])


if __name__ == '__main__':
    tc_save_420k()