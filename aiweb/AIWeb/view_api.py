from AIWeb.utils import *
from AIWeb.view_basic import *
from AIWeb.model import *
def _Response(x: str):
    return HttpResponse(x.encode('utf-8'))
def _Int(dic, k, dft):
    if k in dic:
        return int(dic[k][0])
    return dft
def _Float(dic, k, dft):
    if k in dic:
        return float(dic[k][0])
    return dft
def _Str(dic, k, dft):
    if k in dic:
        return str(dic[k][0])
    return dft
def API_pos(request):
    r'''add predicted position'''
    dict_get = dict(request.GET)
    if 'mac' not in dict_get:
        return _Response('[ERROR] which mac address?')
    if 'x' not in dict_get or 'y' not in dict_get:
        return _Response('[ERROR] what x, y?')
    mac_get = _Str(dict_get, 'mac', 'nopi')
    tag_get = _Str(dict_get, 'tag', '')
    x_get = _Float(dict_get, 'x', 0)
    y_get = _Float(dict_get, 'y', 0)
    print('--------==== API_pos ====--------')
    find_dup = PosPredict.objects.filter(mac=mac_get, tag=tag_get)
    if len(find_dup)>=1:
        toadd = find_dup[0]
        print('dup!', toadd)
        toadd.x = x_get
        toadd.y = y_get
        toadd.save()
        return _Response('modified')
    toadd = PosPredict(
        mac=mac_get, 
        tag=tag_get,
        x=x_get,
        y=y_get,)
    toadd.save()
    return _Response('done')

def API_raw(request):
    dict_get = dict(request.GET)
    if 'pi' not in dict_get:
        return _Response('[ERROR] which pi?')
    if 'c' not in dict_get:
        return _Response('[ERROR] what c? (csv content?)')
    toadd = RawData(
        pi=_Str(dict_get, 'pi', 'nopi'), 
        target=_Str(dict_get, 'c', 'empty'))
    toadd.save()
    return _Response('done')
def API_parsed(request):
    return_str = ''
    dict_get = dict(request.GET)
    print(dict_get)
    maxn = _Int(dict_get, 'maxn', 10)
    lod = []
    for raw_data in RawData.objects.order_by('-created')[:maxn]:
        lod.append(raw_data.to_dict())
    return _Response(json.dumps(lod, indent=1))

def ExampleParser(s0):
    return s0
