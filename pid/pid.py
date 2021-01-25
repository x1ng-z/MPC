# import start
import numpy as np


# {
#     "IN1":{
#     "vib_f":{"value":1}
#     }
#
#
# }
### customer code start
def main(input_data, context):
    IN1 = input_data["IN1"]

    mv = IN1['mv']['value']
    pv = IN1['pv']['value']
    sp = IN1['sp']['value']
    mvup = IN1['mvup']['value']
    mvdown = IN1['mvdown']['value']
    ff=IN1['ff']['value'] if 'ff' in IN1 else 0
    kp = IN1['kp']['value']
    ki = IN1['ki']['value']
    kd = IN1['kd']['value']
    if 'ff' not in context:
        context['ff']=ff
    err = sp - pv
    if 'err_list' not in context:
        context['err_list'] = [err]
    else:
        context['err_list'].append(err)
    if len(context['err_list']) >= 3:
        context['err_list'] = context['err_list'][-3:]
    if len(context['err_list']) == 3:
        delta_u = kp * (context['err_list'][-1] - context['err_list'][-2]) + ki * context['err_list'][-1] + kd * (
                    context['err_list'][-1] - 2 * context['err_list'][-2] + context['err_list'][-3])
    else:
        delta_u = 0

    delta_ff=ff-context['ff']
    #update ff
    context['ff']=ff
    delta_u+=delta_ff

    update_mv=delta_u+mv

    if update_mv>=mvup:
        update_mv=mvup
    elif update_mv<=mvdown:
        update_mv=mvdown

    outpin=input_data['OUT1']
    outpinName=outpin[outpin.keys()[0]]['pinName']

    OUT1 = {
        outpinName:{
            'value':update_mv
        }
    }
    return OUT1