

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from keras.models import load_model
import lightgbm as lgb
import math

from scipy import signal
from scipy.integrate import odeint
from scipy.optimize import minimize
from time import time,sleep
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
pd.set_option('max_columns',300)
pd.set_option('max_row',50)

# ====================================================================
def get_dict(dict_pd):
    comment_dict={}
    for i in range(dict_pd.shape[0]):
        comment_dict[dict_pd.iloc[i]['id']]=dict_pd.iloc[i]['features']
    return comment_dict

# 实时数据预处理
def get_real_time_data(data,comment_dict):
    # 修改第一列列名
    data_columns = list(data.columns)
    data_columns[0] = "time_tab"
    data.columns = data_columns
    # 构造中文名df
    new_columns=[]
    for i in data.columns:
        if i in comment_dict.keys():
            new_columns.append(comment_dict[i])
        else:
            new_columns.append(i)
    data.columns=new_columns

    data['分解炉出口CO'][data['分解炉出口CO']<0] = 0
    data['分解炉出口NOx'][data['分解炉出口NOx']<100] = 100
    data['分解炉出口氧含量'][data['分解炉出口氧含量']<0] = 0
    data['分解炉出口氧含量'][data['分解炉出口氧含量']>10] = 10
    # 转成1min采样
    temp = [pd.DataFrame(data[i:i+12].mean()).T for i in range(0,data.shape[0],12)]
    result = pd.concat(temp)
    return result

def clean_data(data):
    b, a = signal.butter(8, 0.2, 'lowpass')
    filtercols = ['2#篦冷机篦室风压','3#篦冷机篦室风压','1#窑主传电流','入窑提升机总电流']
    for col in filtercols:
        data[col] = signal.filtfilt(b, a, data[col])
    data['分解炉出口CO'] = data['分解炉出口CO']*100
    # 21min反吹1次，取21min平均好了；以后考虑取21min平均值填充一整段数据
    data['分解炉出口CO'] = data['分解炉出口CO'].rolling(21).mean()
    data['分解炉出口NOx'] = data['分解炉出口NOx'].rolling(21).mean()
    data['分解炉出口氧含量'] = data['分解炉出口氧含量'].rolling(21).mean()
    data = data.interpolate(method='linear', limit_direction='forward', axis=0)
    data = data.interpolate(method='linear', limit_direction='backward', axis=0)
    return data

def prepare_data(data,ori_cols,features,yname):
    dfcopy = copy.deepcopy(data)
    for col in ori_cols:
        name = col+"_diff"
        dfcopy[name] = dfcopy[col].diff()
    dfcopy['y'] = dfcopy[yname]
    dfcopy.dropna(inplace=True)
    dfcopy = dfcopy[features]
    return dfcopy

# ====================================================================
# 斐波那契数列
def fib_fun(n):
    if n <= 3:
        return n
    else:
        return fib_fun(n - 1) + fib_fun(n - 2)

def series_to_predict(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    fib_list = []
    for i in range(1,n_in+1):
        fib_list.append(fib_fun(i)-1)
    #
    for i in fib_list[::-1]:    
        if i == 0:
            cols.append(df.iloc[:,:-1].shift(-i))
            names += [('%s(t-%d)min' % (data.columns[j], i+1)) for j in range(n_vars-1)]
        else:
            # 1min采样,间隔1分钟shift_1次
            cols.append(df.iloc[:,:12].shift(i))
            names += [('%s(t-%d)min' % (data.columns[j], i+1)) for j in range(12)]
    
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

def prepare_response_K(data_read_block,datarows):
    data_read_last = pd.DataFrame(data_read_block.mean()).T
    data_read_last = round(data_read_last,2)
    rows = list()
    for i in range(datarows):
        rows.append(data_read_last)
    data_seq = pd.concat(rows)

    # 构造原始序列完毕
    # ==============================
    ori_cols = data_seq.columns
    df_cur = prepare_data(data_seq,ori_cols,features_cur,['1#窑主传电流_diff'])
    df_temp = prepare_data(data_seq,ori_cols,features_temp,['窑头罩红外温度_diff'])

    series_df_cur = series_to_predict(df_cur,n_lag,n_seq)
    series_df_temp = series_to_predict(df_temp,n_lag,n_seq)

    curr_dy = model_cur.predict(series_df_cur.values)[0]
    temp_dy = model_temp.predict(series_df_temp.values)[0]

    mv = data_seq['窑头转子秤喂煤量反馈'].iloc[-1] + 1
    
    return data_seq,curr_dy,temp_dy,mv,ori_cols
    

def get_response_K(data_seq,dy,yname,dyname,mv,ori_cols,features,N,T,tau):
    dy_list = [dy]
    data_read_copy = copy.deepcopy(data_seq)
    # 窑电流的时序固定15min，滞后时间固定1min
    for i in range(N):
        pv = data_read_copy[yname].iloc[-1] + dy
        data_read_last = data_read_copy.iloc[-1,:]
        data_read_last[yname] = pv
        data_read_last['窑头转子秤喂煤量反馈'] = mv
        data_read_copy = data_read_copy.append(data_read_last).iloc[1:,:]
        data_read_copy_copy = copy.deepcopy(data_read_copy)

        df_copy = prepare_data(data_read_copy_copy,ori_cols,features,dyname)
        series = series_to_predict(df_copy,n_lag,n_seq)

        dy = model_cur.predict(series.values)[0]
        dy_list.append(dy)

    # 1h更新一次响应曲线
    dy_sum_list = [sum(dy_list[:i]) for i in range(len(dy_list))]
    K = math.ceil(np.max(dy_sum_list[:(4*T+tau)]))
    return K

# ====================================================================
class response_model():
    def __init__ (self,K,T,tau,M,N,Ts):
        self.K = K
        self.T = T
        self.tau = tau
        self.M = M
        self.N = N
        self.Ts = Ts
    
    def process_model(self,y,t,u,K,T):
        # arguments
        #  y   = outputs
        #  t   = time
        #  u   = input value
        #  K   = process gain
        #  tao = process time constant
        dydt = (-y + self.K * u)/(self.T)
        return dydt
    
    #阶跃响应序列
    def his_res_seq(self):
        t_res = [i * self.Ts for i in range(self.N)]
        res_single=odeint(self.process_model,0,t_res[:self.N-int(self.tau/self.Ts)],args=(1,self.K,self.T)).T[0]
        theta_arr=np.zeros(int(self.tau/self.Ts))
        res_all=np.concatenate([theta_arr,res_single])
        return res_all
    
    #阶跃响应矩阵
    def response_vec(self):
        vec_result = np.array([])
        res_seq = self.his_res_seq()
        for i in range(self.M):
            vec_result_mid = np.concatenate([np.zeros(i),res_seq[:len(res_seq)-i]])
            vec_result = np.append(vec_result,vec_result_mid)
        return vec_result.reshape(self.M,len(res_seq))

def funnel(set_point,P,N,f_up1=5,f_down1=5,f_up2=5,f_down2=5):
    sp_seq = np.zeros((P,N))+set_point
    # 构建一个漏斗
    start_val = np.array([[f_up1],[f_down1]])
    end_val = np.array([[f_up2],[f_down2]])
    funnel_up_start_val = set_point+start_val+end_val
    funnel_up_end_val = set_point+end_val
    funnel_down_start_val = set_point-start_val-end_val
    funnel_down_end_val = set_point-end_val
    funnel_up = np.zeros((P,N))
    funnel_down = np.zeros((P,N))
    for i in range(P):
        funnel_up[i] = np.linspace(funnel_up_start_val[i],funnel_up_end_val[i],N)
        funnel_down[i] = np.linspace(funnel_down_start_val[i],funnel_down_end_val[i],N)
    return funnel_up,funnel_down

# 滚动优化
def objective(u_hat_seq_all,y_hat,response_vec_result,funnel_up,funnel_down,P,M,N,Q,R):
    y_hat0 = np.zeros((P,N)) # 每次迭代不能让y_hat被覆盖
    coeff = np.array([1 - 0.8 ** i for i in range(1,N+1)])
    u_hat_seq_all = u_hat_seq_all.reshape(1,M)
    dy_hat = []
    for i in range(P):
        dy_hat.append(np.dot(u_hat_seq_all,response_vec_result[i]))
    dy_hat = np.array(dy_hat).reshape(P,N)
    y_hat0 += dy_hat    up_seq_diff = funnel_up - y_hat0

    y_hat0 = y_hat0 + y_hat
    down_seq_diff = y_hat0 - funnel_down
    up_seq_diff[up_seq_diff > 0] = 0
    down_seq_diff[down_seq_diff > 0] = 0
    sp_seq_diff = (up_seq_diff+down_seq_diff)* coeff
    result = ((Q*(sp_seq_diff*sp_seq_diff).sum(axis=1)).sum())/1000 + (R*(u_hat_seq_all * u_hat_seq_all)).sum()
    return result

# 反馈矫正
def re_correct(y_pre,y_real,y_hat):
    dif_y = y_real-y_pre
    h_arr = np.ones((P,N)) * 0.5
    y_hat += dif_y * h_arr
    return y_hat


# 通讯校验
def opc_tag(i_tag):
    #i_tag = 0
    if i_tag == 0:
        i_tag = 1
    else :
        i_tag = 0
    return i_tag

# ====================================================================
if __name__ == "__main__":
    
    # 读取实时数据处理config
    features_cur = pd.read_csv("../data/features_cur.csv")# 窑电流特征
    features_temp = pd.read_csv("../data/features_temp2.csv")# 二次风温特征
    # 进行shift前的特征
    features_cur = list(features_cur['feas'])
    features_temp = list(features_temp['feas'])
    # 原始特征字典
    dict_pd = pd.read_csv("../data/dict_features.csv",encoding="utf-8")
    comment_dict = get_dict(dict_pd)
    # model
    model_cur = lgb.Booster(model_file="../model/lgb_delta_current_20200516.h5")
    model_temp = lgb.Booster(model_file="../model/lgb_delta_temp2_20200516.h5")

    n_lag = 8
    n_seq = 1
    datarows = fib_fun(n_lag)+1
    
    # 喂煤模型参数  MV
    T1 = 3
    T2 = 5
    tau1 = tau2 = 1
    T = max(T1,T2)
    tau = max(tau1,tau2)
    P = 2
    M = 3
    Ts = 1
    N = int((4*T+tau)/Ts)
    Q = np.array([1,1])
    R = 200
 
    # 增量限制
    dmv_limit = 0.5
    dmv_high = dmv_limit
    dmv_low = -dmv_limit

    # ====================================================================
    # 需要从外界读取 y_real,set_point,mv_real,mv_high,mv_low
    # 约束规则
    n = 0 # 用于更新模型K,T,tau
    flag = 0 # 用于每次输出头煤后的延时标记
    count = 0 # 用于每次输出头煤后的延时标记
    
    time_tab = ""
    time_mid = 1# 测试用
    dmv_total = 0
    
    while True:
#     while (n < 6):
        try:
            total_starttime = time()
            print("n:",n)
            # ====================================================================
            # 计算响应曲线
            # 10min控制周期,1min采样
            # 循环读取实时值，每min读取一次？
            data_read = pd.read_csv("C:/Users/Administrator/Desktop/opcread/kilncoal.csv",encoding = 'ISO-8859-1')
#             data_read = pd.read_csv("../kilncoal.csv",encoding = 'ISO-8859-1')
            result = get_real_time_data(data_read,comment_dict)
            result = clean_data(result)
            data_read_block = copy.deepcopy(result)
            data_read_block = data_read_block.drop(['time_tab'],axis = 1)

            # 1h更新一次响应曲线
            if (n%60 == 0):
                n = 0
                data_seq,curr_dy,temp_dy,mv,ori_cols = prepare_response_K(data_read_block,datarows)
                K1 = get_response_K(data_seq,curr_dy,'1#窑主传电流','1#窑主传电流_diff',mv,ori_cols,features_cur,N,T1,tau1)
                K2 = get_response_K(data_seq,temp_dy,'窑头罩红外温度','窑头罩红外温度_diff',mv,ori_cols,features_temp,N,T2,tau2)

                print("更新 K1:%s K2:%s "%(K1,K2))

        # ====================================================================

            # 控制部分
            # 响应序列与响应矩阵初始化
            # 若K小于0，则启用规则
            if K1 <= 0:
                K1 = 0
            if K2 <= 0:
                K2 = 0

            res_model_0 = response_model(K1*3,T1,tau1,M,N,Ts)
            res_model_1 = response_model(K2*3,T2,tau2,M,N,Ts)
            res_seq_0 = res_model_0.his_res_seq()
            res_seq_1 = res_model_1.his_res_seq()

            res_vec_0 = res_model_0.response_vec()
            res_vec_1 = res_model_1.response_vec()

            du0 = 0
            dy_hat0 = np.array([du0 * res_seq_0,du0 * res_seq_1])
            response_vec_result = np.array([res_vec_0,res_vec_1])# 多个控制步的响应矩阵

            time_mid += 1 # 测试用

            y_real_0 = data_read_block['1#窑主传电流'].iloc[-1]
            y_real_1 = data_read_block['窑头罩红外温度'].iloc[-1]
            y_real_0 = round(y_real_0,1)
            y_real_1 = round(y_real_1,1)
            mv_read_last_0 = data_read_block['窑头转子秤喂煤量反馈'].iloc[-1]
            mv_read_last_0 = round(mv_read_last_0,1)

            # 1. 头尾煤输出比例约束
            mvh = data_read_block['窑尾转子秤喂煤量'].iloc[-1]*4.5/5.5 + 3
            mv_high_0 = round(mvh)
            mvl = data_read_block['窑尾转子秤喂煤量'].iloc[-1]*3.5/6.5 + 3
            mv_low_0 = round(mvl)

            data_config = pd.read_csv("C:/Users/Administrator/Desktop/opcread/config.csv",encoding = 'ISO-8859-1')
#             data_config = pd.read_csv("../config.csv",encoding = 'ISO-8859-1')
            set_point_get_0 = data_config['APC_Current_SP.VALUE'].iloc[-1]
            set_point_get_1 = data_config['APC_Gcool_SP4.VALUE'].iloc[-1]
            apc_auto = data_config['APC_761RS03_AUTO.VALUE'].iloc[-1]

            if time_tab == "":
                y_start_0 = y_real_0
                y_start_1 = y_real_1
                y_start = np.array([[y_real_0],[y_real_1]])
                y_hat = y_start+dy_hat0
                y0 = y_start

                set_point_0 = set_point_get_0
                set_point_1 = set_point_get_1

                mv_read_last_last_0 = mv_read_last_0
                time_tab = time_mid

                set_point = np.array([[set_point_get_0],[set_point_get_1]])

                apc_auto_last = apc_auto

            if time_tab <= time_mid:
                time_tab += 1 # 测试用
                apc_auto_last = apc_auto
                dmv_read_0 = mv_read_last_0 - mv_read_last_last_0
                mv_read_last_last_0 = mv_read_last_0
                set_point = np.array([[set_point_get_0],[set_point_get_1]])

                if apc_auto == 0.0:
                    print("-----------------------mpc is shut down -------------------\n")
                    mv_out_0 = mv_read_last_0
                    if (n % 10 == 0):
                        pd.DataFrame([mv_out_0],columns=['761RS03.PV']).to_csv("write.csv",index=False)

                if apc_auto == 1.0:
                    print(" 窑电流设定值: %s \n 二次风温设定值: %s \n mv_high is :%s \n mv_low is : %s\n mv0_last:%s"
                          %(set_point_get_0,set_point_get_1,mv_high_0,mv_low_0,mv_read_last_0))
                    print("================")
                    print("y_real_0 is %s"%y_real_0)
                    print("y_real_1 is %s"%y_real_1)
                    print(" set_point: %s \n dmv_read_0 is : %s\n mv_read_last_last_0:%s"
                          %(set_point,dmv_read_0,mv_read_last_last_0))
                    print("================")

                    res_last = np.array([dmv_read_0 * res_seq_0,dmv_read_0 * res_seq_1])
                    for i in range(P):
                        y_hat[i] = np.insert(y_hat[i],len(y_hat[i]),y_hat[i][-1])[1:] + res_last[i]

                    y_real =  np.array([[y_real_0],[y_real_1]])
                    print("实际y与预测y 差值："+str(y_real -  y0))

                    y_hat = re_correct(y0,y_real,y_hat)  ##矫正上一时刻数值
                    print("y pre seq is : %s"%y_hat)

                    y0[0] = y_hat[0][0]
                    y0[1] = y_hat[1][0]
                    print("y0 is : %s"%y0)

                    cons = ({'type':'ineq',
                        'fun':lambda x:np.array(mv_high_0-mv_read_last_0-x[0])},
                       {"type":'ineq',
                        'fun':lambda x:np.array(mv_high_0-mv_read_last_0-x[1])},
                       {"type":'ineq',
                        'fun':lambda x:np.array(mv_high_0-mv_read_last_0-x[2])},
                        {"type":'ineq',
                        'fun':lambda x:np.array(mv_read_last_0+x[0]-mv_low_0)},
                        {"type":'ineq',
                        'fun':lambda x:np.array(mv_read_last_0+x[1]-mv_low_0)},
                        {"type":'ineq',
                        'fun':lambda x:np.array(mv_read_last_0+x[2]-mv_low_0)}                
                       )  

                    bnds = [[dmv_low,dmv_high] for i in range(1 * M)]
                    for m in range(M):
                        if mv_high_0 - mv_read_last_0 >= 0 and mv_high_0 - mv_read_last_0 <= dmv_limit:
                            bnds[m][1] = mv_high_0 - mv_read_last_0
                        if mv_read_last_0 - mv_low_0 >= 0 and mv_read_last_0 - mv_low_0 <= dmv_limit:
                            bnds[m][0] =  mv_low_0 - mv_read_last_0

                    funnel_up,funnel_down = funnel(set_point,P,N,5,5,3,3)
                    solution = minimize(objective,x0=np.zeros((1,M)),args=(y_hat,response_vec_result,funnel_up,funnel_down,P,M,N,Q,R)
                                        ,bounds=bnds,constraints=cons)
                    print("约束求解正常？%s"%solution.success)
                    print("delta_x:"+str(solution.x))
                    out_put_0 = solution.x[0]
                    dmv_total = dmv_total+out_put_0

                # 添加约束
                # 2. 每次输出之后等待10min-15min，期间还要继续累积增量
                if ((np.abs(dmv_total) >= 0.2) & (flag == 0)):
                    print("本次输出mv：")
                    if dmv_total > dmv_high:
                        dmv_total = dmv_high
                    elif dmv_total < dmv_low:
                        dmv_total = dmv_low
                    mv_out_0 = mv_read_last_0 + dmv_total
                    print("delta_x:"+str(dmv_total))
                    mv_out_0 = round(mv_out_0,1)
                    print("mv_out_0:",mv_out_0)
                    flag = 1
                    count = 0

                    pd.DataFrame([mv_out_0],columns=['761RS03.PV']).to_csv("write.csv",index=False)
                    dmv_total = 0

                del data_read_block
                del data_read
                del result
                del data_config

                # ====================================================================
                # 多久计算一次响应序列，多久控制一次，都用10min
                total_endtime = time()
                print("total time:",total_endtime - total_starttime)
                print("===============================")
                if 60-(total_endtime - total_starttime) > 0:
                    sleep(60-(total_endtime - total_starttime))
                n += 1
                count += 1
                if count >= 10:
                    flag = 0
        except:
            pass
            print("error")
    