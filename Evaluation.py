import scipy.stats as st
import numpy as np
import pandas as pd

def l1_distance(imputed_data, original_data):

    return np.mean(np.abs(original_data-imputed_data))
    
def RMSE(raw, impute, scale = 'zscore'):

    def scale_z_score(df):
        result = pd.DataFrame()
        for label, content in df.items():
            content = st.zscore(content)
            content = pd.DataFrame(content,columns=[label])
            result = pd.concat([result, content],axis=1)
        return result
    
    if scale == 'zscore':
        raw = scale_z_score(raw)
        impute = scale_z_score(impute)
    else:
        print ('Please note you do not scale data by zscore')
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                RMSE = 1.5   
            else:
                raw_col =  raw.loc[:,label]
                impute_col = impute.loc[:,label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                RMSE = np.sqrt(((raw_col - impute_col) ** 2).mean())

            RMSE_df = pd.DataFrame(RMSE, index=["RMSE"],columns=[label])
            result = pd.concat([result, RMSE_df],axis=1)
            mean_values = result.mean().mean()
    else:
        print("columns error")
    return mean_values     

def SPCC(raw, impute, scale=None):
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                spearmanr = 0
            else:
                raw_col = raw.loc[:, label]
                impute_col = impute.loc[:, label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                spearmanr, _ = st.spearmanr(raw_col, impute_col)
            spearman_df = pd.DataFrame(spearmanr, index=["SPCC"], columns=[label])
            result = pd.concat([result, spearman_df], axis=1)
            mean_values = result.mean().mean()
    else:
        print("columns error")
    return mean_values

def cal_ssim(im1,im2,M):
    assert len(im1.shape) == 2 and len(im2.shape) == 2
    assert im1.shape == im2.shape
    mu1 = im1.mean()
    mu2 = im2.mean()
    sigma1 = np.sqrt(((im1 - mu1) ** 2).mean())
    sigma2 = np.sqrt(((im2 - mu2) ** 2).mean())
    sigma12 = ((im1 - mu1) * (im2 - mu2)).mean()
    k1, k2, L = 0.01, 0.03, M
    C1 = (k1*L) ** 2
    C2 = (k2*L) ** 2
    C3 = C2/2
    l12 = (2*mu1*mu2 + C1)/(mu1 ** 2 + mu2 ** 2 + C1)
    c12 = (2*sigma1*sigma2 + C2)/(sigma1 ** 2 + sigma2 ** 2 + C2)
    s12 = (sigma12 + C3)/(sigma1*sigma2 + C3)
    ssim = l12 * c12 * s12
    
    return ssim
    
def SSIM(raw, impute, scale = 'scale_max'):
    
    def scale_max(df):
        result = pd.DataFrame()
        for label, content in df.items():
            content = content/content.max()
            result = pd.concat([result, content],axis=1)
        return result
    
    if scale == 'scale_max': # large time consumer
        raw = scale_max(raw)
        impute = scale_max(impute)
    else:
        print ('Please note you do not scale data by scale max')
    # print(raw.shape, impute.shape) # (92614, 1890)
    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                ssim = 0
            else:
                raw_col =  raw.loc[:,label]  # return 'label' column
                impute_col = impute.loc[:,label]
                impute_col = impute_col.fillna(1e-20)
                raw_col = raw_col.fillna(1e-20)
                M = max(raw_col.max(), impute_col.max())
                raw_col_2 = np.array(raw_col)
                raw_col_2 = raw_col_2.reshape(raw_col_2.shape[0],1)
                impute_col_2 = np.array(impute_col)
                impute_col_2 = impute_col_2.reshape(impute_col_2.shape[0],1)
                ssim = cal_ssim(raw_col_2,impute_col_2,M)
            
            ssim_df = pd.DataFrame(ssim, index=["SSIM"],columns=[label])
            result = pd.concat([result, ssim_df],axis=1)
            mean_values = result.mean().mean()
    else:
        print("columns error")
    return mean_values

def JS(raw, impute, scale='scale_plus'):

    def scale_plus(df):
        result = pd.DataFrame()
        for label, content in df.items():
            if content.sum() == 0:
                content = content.fillna(1e-20)
            else:
                content = content / content.sum()
            result = pd.concat([result, content], axis=1)
        return result

    def safe_expm1(df):
        max_value = 1000
        return np.clip(df, a_min=None, a_max=max_value).apply(np.expm1)

    if scale == 'scale_plus':
        raw = scale_plus(safe_expm1(raw))
        impute = scale_plus(safe_expm1(impute))
    else:
        print('Please note you do not scale data by plus')

    if raw.shape[0] == impute.shape[0]:
        result = pd.DataFrame()
        for label in raw.columns:
            if label not in impute.columns:
                JS = 1
            else:
                raw_col = raw.loc[:, label]
                impute_col = impute.loc[:, label]
                raw_col = raw_col.fillna(1e-20)
                impute_col = impute_col.fillna(1e-20)
                M = (raw_col.to_numpy() + impute_col.to_numpy()) / 2
                t1 = st.entropy(raw_col, M)
                t2 = st.entropy(impute_col, M)
                JS = (t1 + t2) / 2

            JS_df = pd.DataFrame(JS, index=["JS"], columns=[label])
            result = pd.concat([result, JS_df], axis=1)

        mean_values = result.mean().mean()
    else:
        print("columns error")
        mean_values = None

    return mean_values