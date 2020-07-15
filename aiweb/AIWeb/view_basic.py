from AIWeb.utils import *
from AIWeb.model import *
_MakoMgr = MakoManager(['mako'])
_Mako = _MakoMgr.render
def get_common_nav():
    return _Mako(r'<%include file="nav.html"/>')

@requires_csrf_token
def basic_view(request):
    dict_get = dict(request.GET)
    dump_content = json.dumps(dict_get, indent=1)
    title = 'basic_view'
    nav = get_common_nav()
    body = '\n'.join([
        f'<pre style="text-align: left;">',
        f'{dump_content}',
        f'</pre>',
    ])
    
    # collect (raw, context)
    raw = '\n'.join([
        r'<%inherit file="basic.html"/>',
    ]).strip()
    context = locals().copy()
    response = HttpResponse(_Mako(raw, context).encode('utf-8'))
    return response

def _Str(dic, k, dft):
    if k in dic:
        return str(dic[k][0])
    return dft

def epy_view(request):
    dict_get = dict(request.GET)
    tag_get = _Str(dict_get, 'tag', '')
    dump_content = json.dumps(dict_get, indent=1)
    init_point_list = [
        [2.7, 2.0, '#00f'],
        [0.5, 13, '#00f'],
        [0.2, 19.4, '#00f'],
        [12.2, 0.5, '#00f'],
        [16, 12.5, '#00f'],
        [16.1, 19, '#00f'],
        [0.0, 0.0, '#000'],
        [-5, 25, '#fee'],
        [20, -5, '#fee'],
        [-5, -5, '#fee'],
        [20, 25, '#fee'],
    ]
    for raw_pos in PosPredict.objects.filter(tag=tag_get):
        di = raw_pos.to_dict()
        mac = di['mac']
        print('================================================================')
        print(di['id'], type(di['id']))
        print('================================================================')
        # r = ord(mac[0]) % 10
        # g = ord(mac[1]) % 10
        r = (di['id'] * 23307 % 5) + 5
        g = (di['id'] * 333071 % 5) + 5
        b = (di['id'] * 13 % 5) + 5
        clr = di['id'] * 1337 % 4095

        # print('mac', di['mac'], r, g, b)
        # init_point_list.append([di['x'], di['y'], f'#cc{b}'])
        # init_point_list.append([di['x'], di['y'], f'#{r}{g}{b}'])
        init_point_list.append([di['x'], di['y'], f'#{clr:03x}'])
        # f'{}'
    
    # prepare body
    title = 'epy_view'
    nav = ''
    body = _Mako(r'<%include file="body_epy.html"/>', {
        'init_point_list': init_point_list
    })
    
    # collect (raw, context) --
    raw = '\n'.join([
        r'<%inherit file="basic.html"/>',
    ]).strip()
    context = locals().copy()
    response = HttpResponse(_Mako(raw, context).encode('utf-8'))
    return response

def test_view(request):
    print("--------==== test_view ====--------")
    with open('AIWeb/epy.htm', 'r') as f:
        s0 = f.read()
    response = HttpResponse(s0.encode('utf-8'))
    return response


# --------==== tring channels ====--------
def chat_view(request):
    print("--------==== chat ====--------")
    with open('mako/chat.html', 'r') as f:
        s0 = f.read()
    response = HttpResponse(s0.encode('utf-8'))
    return response



def test_sca_view(request):
    # prepare body
    title = 'epy_view'
    nav = ''
    body = _Mako(r'<%include file="body_epy_sca.html"/>', {
        'init_point_list': []
    })
    
    # collect (raw, context) --
    raw = '\n'.join([
        r'<%inherit file="basic.html"/>',
    ]).strip()
    context = locals().copy()
    response = HttpResponse(_Mako(raw, context).encode('utf-8'))
    return response