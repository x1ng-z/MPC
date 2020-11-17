import numpy as np
# import matplotlib.pyplot as plt
import apc
import Help
import sys
import requests
import json
import time

import gc
import os
import sys
import traceback


class OrigionAPC:
    def __init__(self,p,m,N):
        self.p=p
        self.m=m
        self.N=N
        self.origion_A_step_response_sequence=[]
        self.origion_B_step_response_sequence=[]
        self.H = np.zeros((p * N, p))  # [输出引脚*阶跃时序长度，输出引脚]

        '''build矫正 H matrix'''
        for indexp in range(p):
            for indexn in range(N):
                if indexn == 0:
                    self.H[indexn + N * indexp, indexp] = 1  # hi[loop_outi]
                else:
                    self.H[indexn + N * indexp, indexp] = 0.7  # hi[loop_outi]

        '''位移矩阵'''
        self.S = np.zeros((p * N, p * N))  # [输出引脚*阶跃时序长度，输出引脚*阶跃时序长度]

        '''构造计算位移矩阵S'''
        for loop_outi in range(p):
            for loop_stepi in range(0, N - 1):
                self.S[loop_outi * N + loop_stepi, loop_outi * N + loop_stepi + 1] = 1
            self.S[loop_outi * N + N - 1, loop_outi * N + N - 1] = 1
        pass

    def setOrigion_A_step_response_sequence(self,origion_A_step_response_sequence):
        self.origion_A_step_response_sequence=origion_A_step_response_sequence

    def getOrigion_A_step_response_sequence(self):
        return self.origion_A_step_response_sequence

    def setOrigion_B_step_response_sequence(self,origion_B_step_response_sequence):
        self.origion_B_step_response_sequence=origion_B_step_response_sequence

    def getOrigion_B_step_response_sequence(self):
        return self.origion_B_step_response_sequence




if __name__ == '__main__':
    try:
        tools = Help.Tools()

        DEBUG = True
        for i in range(1, len(sys.argv)):
            strs = sys.argv[i]
            print(i)
            print(strs)
            # stringjson = '{"p":1,"P":12,"1":[{"3":[0.0,0.4151304360067797,1.176822478429792,1.590779306025315,1.4264776926660865,0.9715664675338392,0.6636080291971198,0.7046978340239161,0.96691428652255,1.1833583291582954,1.1962406597511275,1.0511694712410666,0.9055599035571009,0.8741424378067684,0.9503684965178477,1.0447759901910296,1.0780947838763422,1.0408348570641193,0.981644839331238,0.9531081048328375,0.9692863400976179,1.0052033436203236,1.0271898224847225,1.0217590754024484,1.0006928920927902,0.9848504554785261,0.9852679605682885,0.9971640670749762,1.0080340643148051,1.0096051384111164,1.0031874978182627,0.9960190291852486,0.9939463573954769,0.9972050203498327,1.0017707203955273,1.0036926839547362,1.0021824697550235,0.999368400179349,0.9978219848878931,0.9984129800783667,1.000090988409581,1.0012378717318327,1.0010965742962357,1.0001310550954547,0.9993268144975218,0.9992725589200009,0.9998055991591668,1.0003456436656775,1.0004658026258857,1.0001865065942093,0.9998369303088337,0.9997114086035236,0.9998473594328934,1.000066195931829,1.0001730129950057,1.0001144306425287,0.9999818434768685,0.9998998531347962,0.9999191234732244,0.9999967599662898,1.0000556852952884,1.0000546539582504,1.0000108968029757,0.9999705476656342,0.9999644247984766,0.9999879735182591,1.000014537281888,1.000022386862253,1.0000104701510673,0.9999935776849916,0.9999863650809752,0.9999918572434927,1.0000022530706456,1.0000080290066224,1.0000059050072585,0.9999997159043338,0.9999954452523759,0.9999959283433218,0.999999483207258,1.0000024712435802,1.000002696333106,1.000000737982725,0.9999987351965977,0.9999982761739459,0.999999299368072,1.0000005937009586,1.0000010663847057,1.000000570463732,0.9999997614425409,0.9999993616980226,0.9999995737163868,1.0000000631244375,1.0000003688338526,1.0000003005815201,1.000000014486805,0.9999997953393841,0.9999997972571651,0.9999999581921387,1.0000001079543899,1.000000131752676]}],"m":1,"M":6}'
        url = 'http://localhost:80/pythonsimulate/modlebuild/86.do'  # sys.argv[1]
        modleId = 86#int(sys.argv[2])
        resp = requests.get(url)  # sys.argv[0]'http://localhost:80/pythonsimulate/modleparam/1.do'
        modle = json.loads(resp.text)
        modlevalidekey = modle['validekey']
        matrixPvMvMapping=np.array(modle['pvmvmapping'])#未拆解之前的映射关系
        matrixPvFfMapping=(np.array(modle["pvffmapping"]) if ("pvffmapping" in modle) else [])##未拆解之前的映射关系
        pvMvEffect=np.array(modle['pvmveffect'])#pv对mv输出的影响占比矩阵

        origionA = np.array(modle['origionA'])  # 原始未拆分响应矩阵(p,m,N)
        Origionapc=OrigionAPC(origionA.shape[0],origionA.shape[1],origionA.shape[2])

        # 变下形状
        Origionapc.setOrigion_A_step_response_sequence(tools.build_A_respond(origionA))
        origionB = (np.array(modle['origionB']) if("origionB" in modle) else [])
        if origionB!=[]:
            Origionapc.setOrigion_B_step_response_sequence(tools.build_B_respond(origionB))

        Y0Position=tools.getFirstY0Position(matrixPvMvMapping)

        MPC = apc.apc(modle["P"], modle["p"], modle["M"], modle["m"], modle["N"], modle["APCOutCycle"], modle["fnum"],
                      np.array(modle["A"]), (np.array(modle["B"]) if ("B" in modle) else []), np.array(modle["Q"]),
                      np.array(modle["R"]), np.array(modle['pvusemv']),np.array(modle['alphe']),np.array(modle['funneltype']),modle['alphemethod'])
        isEnable = modle["enable"]
        resp_opc = requests.get("http://localhost:80/pythonsimulate/opcread/%d.do" % modleId)
        modle_init_data = json.loads(resp_opc.text)
        y0 = np.zeros((MPC.p * MPC.N, 1))
        for indexp in range(MPC.p):
            for indexn in range(MPC.N):
                y0[indexp * MPC.N + indexn, 0] = np.array(modle_init_data['y0'])[indexp]

        ypv=np.array(modle_init_data['y0'])
        limitmv = np.array(modle_init_data['limitU'])#拆分映射的mv上下限
        origionstructlimitmv=np.array(modle_init_data['origionstructlimitmv'])#原始结构的mv上下限
        limitdmv = np.array(modle_init_data['limitDU'])  # ([[0.1,0.2],[0.1,0.2]])#拆分映射的dmv上下限
        origionstructlimitDmv=np.array(modle_init_data['origionstructlimitDmv'])#原始结构的dmv上下限
        mv = np.array(modle_init_data['U'])#原始结构的mv
        origionstructmv=np.array(modle_init_data['origionstructmv'])#原始结构的mv
        mvfb = np.array(modle_init_data['UFB'])#原始结构的fbmv
        originstructmvfb=np.array(modle_init_data['originstructmvfb'])#原始结构的fbmv
        ff = np.array(modle_init_data['FF']) if ('FF' in modle_init_data) else []  # 前馈值
        ffdependregion = np.array(modle_init_data['FFLmt']) if (
                'FFLmt' in modle_init_data) else []  # 前馈置信区间，不在这个区间内的ff,不可以用
        wp = np.array(modle_init_data['wi'])
        deadZones = np.array(modle_init_data['deadZones'])
        funelInitValues = np.array(modle_init_data['funelInitValues'])
        realvalidekey = modle_init_data['validekey']

        isEnable = modle_init_data['enable']
        while ((isEnable == 0) and (realvalidekey == modlevalidekey)):

            originalfunnels, decoratefunnels = tools.buildFunel(wp, deadZones, funelInitValues, MPC.N, MPC.p,MPC.funneltype,MPC.PINF,MPC.NINF)
            '''
            comstraindmv计算出的多部输出的dmv矩阵(m*M,1) 
            firstdmvs是经过约束的(结果就是dmv+mv不能超过上下限，dmv增量不能超过上下限/低于下限则直接变成0)
            originfristdmv是经过算法算出来的源生dmv，未经过任何约束处理的，本次要作用给mv的增量
            '''
            comstraindmv, firstdmvs, originfristdmv = MPC.rolling_optimization(wp, y0, ypv, mv, limitmv, limitdmv, decoratefunnels)
            #predicty0 = MPC.predictive_control(y0, comstraindmv)
            '''新增加死区时间和漏斗初始值'''
            writemv = []

            '''这里说明下，循环遍历每一个pv的漏斗
            如果出现不符合的则需要更新对应影响他的mv
            整理需要用上映射关系
            '''
            # updatemvmat = np.zeros(MPC.m)  # 需改更新的mv
            # for indexp in range(MPC.p):
            #     if ((decoratefunnels[0, indexp * MPC.N:(indexp + 1) * MPC.N] >= y0[indexp * MPC.N:(indexp + 1) * MPC.N,
            #                                                                     0]).all() and (
            #             decoratefunnels[1, indexp * MPC.N:(indexp + 1) * MPC.N] <= y0[indexp * MPC.N:(indexp + 1) * MPC.N,
            #                                                                        0]).all()):
            #         pass
            #     else:
            #         updatemvmat = updatemvmat + MPC.pvusemv[indexp]
            # selectmvmat = updatemvmat > 0  # 大于0说明PV不在漏斗内，需要更新
            # writemv = mv + firstdmvs.reshape(-1) * selectmvmat.astype(int).reshape(-1)

            '''
            分配dmv到响应位置上去'''
            dmv_structe = matrixPvMvMapping.copy()
            dmv_structe=dmv_structe.astype(np.float64)
            dmv_structe[dmv_structe==1]=originfristdmv.reshape(-1)

            #计算本次个dmv的增量

            tmp_dmv_ratio=pvMvEffect*dmv_structe
            UnCheckLimitDmv=np.sum(tmp_dmv_ratio,axis=0)/np.sum(pvMvEffect.astype(np.float64),axis=0)
            UnCheckLimitDmv = UnCheckLimitDmv.reshape(-1, 1)
            #检测本次dmv的限制对不对
            origionstruct_firstonedmv, origionstruct_origfristonedmv=MPC.origionmvconstraint(origionstructmv, UnCheckLimitDmv, origionstructlimitmv, origionstructlimitDmv)

            writemv=origionstructmv+origionstruct_firstonedmv.reshape(-1)
            # writemv =firstdmvs.reshape(-1) * selectmvmat.astype(int).reshape(-1)
            payload = {'id': modleId, 'U': writemv.tolist(), 'validekey': modlevalidekey}
            write_resp = requests.post("http://localhost:80/pythonsimulate/opcwrite.do", params=payload)
            if DEBUG:
                print(write_resp.text)

            # 这里要改成输出周期结束以后再取读取反馈值
            time.sleep((MPC.outStep - MPC.costtime) if ((MPC.outStep - MPC.costtime) >= 0) else 0)

            resp_opc = requests.get("http://localhost:80/pythonsimulate/opcread/%d.do" % modleId)
            modle_real_data = json.loads(resp_opc.text)
            isEnable = modle_real_data["enable"]

            #这里要改为y0收到多个mv共同作用时候所表现出的增量叠加
            # e, y_0N, _dff= MPC.feedback_correction(np.array(modle_real_data['y0']), y0, mvfb, np.array(modle_real_data["UFB"]),
            #                                   ff, (np.array(modle_real_data["FF"]) if ('FF' in modle_real_data) else []),
            #                                   (np.array(modle_real_data["FFLmt"]) if ('FFLmt' in modle_real_data) else []))

            e, y_0N,y_0send, _dff,funnel_send = MPC.feedback_correction_for_simulate(Origionapc,
                                                                 matrixPvMvMapping,
                                                                 Y0Position,
                                                                 np.array(modle_real_data['origiony0']),
                                                                 y0,
                                                                 originstructmvfb,
                                                                 np.array(modle_init_data['originstructmvfb']),
                                                                 ff,
                                                                 (np.array(modle_real_data["FF"]) if ('FF' in modle_real_data) else []),
                                                                 (np.array(modle_real_data["FFLmt"]) if ('FFLmt' in modle_real_data) else []),
                                                                originalfunnels
                                                                 )
            y0 = y_0N

            # limitmv = np.array(modle_real_data['limitU'])#mv限制
            # limitdmv = np.array(modle_real_data['limitDU'])#dmv限制
            # mv = np.array(modle_real_data['U'])  # mv值
            # mvfb = np.array(modle_real_data['UFB'])  # mv反馈
            limitmv = np.array(modle_init_data['limitU'])  # 拆分映射的mv上下限
            origionstructlimitmv = np.array(modle_init_data['origionstructlimitmv'])  # 原始结构的mv上下限
            limitdmv = np.array(modle_init_data['limitDU'])  # ([[0.1,0.2],[0.1,0.2]])#拆分映射的dmv上下限
            origionstructlimitDmv = np.array(modle_init_data['origionstructlimitDmv'])  # 原始结构的dmv上下限
            mv = np.array(modle_init_data['U'])  # 原始结构的mv
            origionstructmv = np.array(modle_init_data['origionstructmv'])  # 原始结构的mv
            mvfb = np.array(modle_init_data['UFB'])  # 原始结构的fbmv
            originstructmvfb = np.array(modle_init_data['originstructmvfb'])  # 原始结构的fbmv



            ff = np.array(modle_real_data['FF']) if ('FF' in modle_real_data) else []  # 前馈值
            ffdependregion = np.array(modle_real_data['FFLmt']) if (
                    'FFLmt' in modle_real_data) else []  # 前馈置信区间，不在这个区间内的ff,不可以用
            wp = np.array(modle_real_data['wi'])  # sp值
            ypv = np.array(modle_real_data['y0'])
            realvalidekey = modle_real_data['validekey']
            '''模型更新退出运算'''
            if (modlevalidekey != realvalidekey):
                break
            payload = {'id': modleId, 'validekey': modlevalidekey
                , 'data': json.dumps(
                    {'dmv': origionstruct_firstonedmv.reshape(-1).tolist()#originfristdmv.reshape(-1).tolist()
                        , 'e': e.reshape(-1).tolist()
                        , 'dff': _dff.reshape(-1).tolist()
                        , 'predict': y_0send.reshape(-1).tolist()
                        , 'funelupAnddown': funnel_send.tolist()
                     }
                )
                       }
            write_resp = requests.post("http://localhost:80/pythonsimulate/updateModleData.do", data=payload)
            if DEBUG:
                print(write_resp.text)
    except:
        traceback.print_exc()
    finally:
        gc.collect()


# pyinstaller -F E:\WX_PUSH_3\WX_Push\WX_Clinet_SalePush.py