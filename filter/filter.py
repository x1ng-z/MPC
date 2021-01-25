#import start
import numpy as np
### customer code start

print(np.array([1,2,3]).mean())

def main(input_data, context):
    OUT = {}
    if(input_data['filterinfo']['filtermethod']=='mvav'):
        if 'yn' not in context:
            yn={}
            for subdata in input_data['data']:
                yn[str(subdata['outputpropertyid'])]={
                'value':[],
                'outputmodleid':subdata['outputmodleid'],
                'outputpropertyid':subdata['outputpropertyid']
                }
            context['yn']=yn
        for subdata in input_data['data']:
            sampledatas=context['yn'][str(subdata['outputpropertyid'])]['value']
            capacity = input_data['filterinfo']['capacity']
            filtevalue=None
            if(len(sampledatas)!=capacity):
                context['yn'][str(subdata['outputpropertyid'])]['value'].append(subdata['value'])
                filtevalue=subdata['value']
            elif(len(sampledatas)==capacity):
                #remove oldest data
                context['yn'][str(subdata['outputpropertyid'])]['value']=context['yn'][str(subdata['outputpropertyid'])]['value'][1:capacity]
                #add newst data
                context['yn'][str(subdata['outputpropertyid'])]['value'].append(subdata['value'])
                filtevalue=np.array(context['yn'][str(subdata['outputpropertyid'])]['value']).mean()

            OUT[str(subdata['outputpropertyid'])]={
                'value':filtevalue ,
                'outputmodleid': subdata['outputmodleid'],
                'outputpropertyid': subdata['outputpropertyid']
            }

        pass
    elif(input_data['filterinfo']['filtermethod']=='fodl'):
        if 'yn' not in context:
            yn={}
            for subdata in input_data['data']:
                yn[str(subdata['outputpropertyid'])]={
                'value':subdata['value'],
                'outputmodleid':subdata['outputmodleid'],
                'outputpropertyid':subdata['outputpropertyid']
                }
            context['yn']=yn

        for subdata in input_data['data']:
            Yn_1=context['yn'][str(subdata['outputpropertyid'])]['value']
            alphe = input_data['filterinfo']['alphe']
            context['yn']['value']=(1 - alphe) * Yn_1 + subdata['value'] * alphe
        OUT=context['yn']
        pass

    return OUT
### customer code end