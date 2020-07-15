


from AIWeb.utils import *
# coova_homepage = 'https://10.1.0.1:3990/www/coova.html'


@requires_csrf_token
def test_view(request):
    print(dir(AIWeb.utils))
    print('AIWeb.utils')
    print("test_view start")
    response = HttpResponse('hello'.encode('utf-8'))
    return response

def Request2Dict(request)->dict:
    return {
        'POST': request.POST,
        'GET': request.GET,
        'COOKIES': request.COOKIES,
        'session': dict(request.session),
    }

def AlreadyLogin(request):
    return request.session['is_login']
# def DBGetUsernameByIP(client_ip):
#     with TrueDB('userinfo.db') as db:
#         x = db.get()
#         obj = x.get('ip_username', {})
#         obj_ip = obj.get(client_ip, {})
#         username = obj_ip.get('username', None)
#     return username
# def _RebuildSessionFromChilliQuery(request):
#     client_ip = get_client_ip(request)
#     print("[session] client_ip = %s"%(client_ip))
    
#     request.session['ip'] = client_ip
#     if not 'is_login' in request.session:
#         request.session['is_login'] = False
#     if not 'username' in request.session:
#         request.session['username'] = None
    
#     lod = ChilliListOfDict()
#     my_row = None
#     for d in lod:
#         if d['IP Address'] == client_ip:
#             my_row = d
#             break
#     request.session['is_free_node'] = type(my_row)==type(None)
#     if request.session['is_free_node']:
#         print("[session] client_ip = %s is_free_node"%(client_ip))
#         return True
#     jprint(my_row)
#     request.session['mac'] = my_row['the MAC Address']
#     request.session['authorized'] = "1" == my_row['authenticated status (1 authorized 0 not)']
#     request.session['acct_session'] = my_row['the session id (used in Acct-Session-ID)']
#     if request.session['authorized']:
#         if type(request.session['username'])==type(None) or not request.session['is_login']:
#             print("[Rebuild] username mending")
#             username = DBGetUsernameByIP(client_ip)
#             if type(username)==type('username'):
#                 request.session['is_login'] = True
#                 request.session['username'] = username
#     else:
#         return _MakeSessionLogout(request)
#     return True

def RebuildSession(request):
    return _RebuildSessionFromChilliQuery(request)

# def _RebuildSessionOldGETWAY(request):
#     print("[session] get_client_ip = %s"%(get_client_ip(request)))
#     coova_list = [
#         "res", "uamip", "uamport",
#         "challenge", "called", "mac",
#         "ip", "nasid",
#         "sessionid",
#         "userurl", "md"]
#     for key in coova_list:
#         if not key in request.GET:
#             return _RebuildSessionFromChilliQuery(request)
#     print(f"[session] rebuild session from chilli GET")
#     for key in coova_list:
#         if key in request.GET:
#             val = request.GET[key]
#             request.session[key] = val
#             print(f"[session] request.session[{key}] =", val)
#     return True

def _MakeSessionLoginPOST(request, assign=None):
    request.session['is_login'] = True
    if assign:
        request.session['username'] = assign
    else:
        request.session['username'] = request.POST['username']
    if request.session['is_free_node']:
        return True
    # with TrueDB('userinfo.db') as db:
    #     x = db.get()
    #     obj = x.get('ip_username', {})
    #     obj_ip = {}
    #     obj_ip['ip'] = request.session['ip']
    #     obj_ip['mac'] = request.session['mac']
    #     # obj_ip['authorized'] = request.session['authorized']
    #     obj_ip['acct_session'] = request.session['acct_session']
    #     obj_ip['username'] = request.session['username']
    #     obj.update({obj_ip['ip']: obj_ip})
    #     x.update({'ip_username': obj})
    #     db.set(x)
    # rc = run('sudo chilli_query authorize ip %s'%(request.session['ip']))
    return rc==0

def _MakeSessionLogout(request):
    request.session['is_login'] = False
    request.session['username'] = None
    if request.session['is_free_node']:
        return True
    # with TrueDB('userinfo.db') as db:
    #     x = db.get()
    #     obj = x.get('ip_username', {})
    #     obj.update({request.session['ip']: {}})
    #     x.update({'ip_username': obj})
    #     db.set(x)
    # rc = run('sudo chilli_query logout ip %s'%(request.session['ip']))
    return rc==0

@never_cache
@requires_csrf_token
def register(request):
    if request.method == 'POST':
        if request.POST['password'] != request.POST['password_again']:
            return redirect('/register/?msg=password_different', permanent=True)
        # with TrueDB('userinfo.db') as db:
        #     x = db.get()
        #     if request.POST['username'] in [x0[0] for x0 in x['user']]:
        #         return redirect('/register/?msg=sameUserNameQAQ', permanent=True)
        #     x['user'].append([request.POST['username'], md5(request.POST['password'])])
        #     db.set(x)
        assert _MakeSessionLoginPOST(request)
        return redirect('/', permanent=True)
    
    print("[register] request = %s"%json.dumps(Request2Dict(request), indent=1))
    RebuildSession(request)
    
    if AlreadyLogin(request):
        return redirect('/', permanent=True)
    
    def _GetPage():
        csrftoken = str(csrf(request)["csrf_token"])
        title = "Register"
        str_session = json.dumps(Request2Dict(request)['session'])
        msg = request.GET.get('msg', '')
        raw = '\n'.join([
            r'<%def name="body()">',
            r'<h3>✏️ 來註冊ㄅ</h3>',
            r'<small id="msgRegister">${msg}</small>',
            r'<%include file="register.html"/>',
            r'</%def>',
            r'',
            r'<%inherit file="basic.html"/>',
        ]).strip()
        return MakoRender(raw, locals().copy())
    print("[register] new session = %s"%json.dumps(dict(request.session), indent=1))
    response = HttpResponse(_GetPage().encode('utf-8'))
    return response


@never_cache
@requires_csrf_token
def login(request):
    if request.method == 'POST':
        with TrueDB('userinfo.db') as db:
            x = db.get()
            tpwd = None
            for uname, pwd in x['user']:
                if request.POST['username']==uname:
                    tpwd = pwd
                    break
            if type(tpwd)==type(None):
                return redirect('/login/?msg=no_such_user', permanent=True)
        if md5(request.POST['password']) == tpwd:
            assert _MakeSessionLoginPOST(request)
            return redirect('/', permanent=True)
        return redirect('/login/?msg=password_wrong', permanent=True)
    
    print("[login] request = %s"%json.dumps(Request2Dict(request), indent=1))
    RebuildSession(request)
    
    if AlreadyLogin(request):
        return redirect('/', permanent=True)
    
    def _GetPage():
        csrftoken = str(csrf(request)["csrf_token"])
        title = "CNLogin"
        str_session = json.dumps(Request2Dict(request)['session'])
        msg = request.GET.get('msg', '')
        raw = '\n'.join([
            r'<%def name="body()">',
            r'<h3>🔌 想上網ㄇ</h3>',
            r'<small id="msgLogin">${msg}</small>',
            r'<%include file="login.html"/>',
            r'<hr>',
            r'<%include file="article.html"/>',
            r'</%def>',
            r'',
            r'<%inherit file="basic.html"/>',
        ]).strip()
        return MakoRender(raw, locals().copy())
    print("[login] new session = %s"%json.dumps(dict(request.session), indent=1))
    response = HttpResponse(_GetPage().encode('utf-8'))
    return response

@never_cache
@requires_csrf_token
def logout(request):
    RebuildSession(request)
    _MakeSessionLogout(request)
    return redirect('/login/', permanent=True)

@never_cache
@requires_csrf_token
def general(request):
    print("[general] request = %s"%json.dumps(Request2Dict(request), indent=1))
    RebuildSession(request)
    if not AlreadyLogin(request):
        return redirect('/login/', permanent=True)
    
    def _GetPage():
        csrftoken = str(csrf(request)["csrf_token"])
        title = "CNLogin"
        raw = '\n'.join([
            r'<%def name="body()">',
            r'<h3>🐧 你已經登入ㄌ, %s</h3>'%request.session['username'],
            r'<a href="/logout/"><button type="button" class="btn btn-danger">Logout</button></a>',
            r'</%def>',
            r'',
            r'<%inherit file="basic.html"/>',
        ]).strip()
        return MakoRender(raw, locals().copy())
    print("[login] new session = %s"%json.dumps(dict(request.session), indent=1))
    response = HttpResponse(_GetPage().encode('utf-8'))
    return response

@never_cache
def adminPage(request):
    def _GetPageDump():
        title = "Admin Page"
        head_msg = ''
        dump = ''
        if settings.DEBUG:
            head_msg = '<h1>🔪 你DEBUG flag沒拿掉ㄛ！！！ 🔪</h1>'
            with TrueDB('userinfo.db') as db:
                x = db.get()
                dump += json.dumps(x, indent=1)
            dump+= '<hr>'
            dump+=json.dumps(ChilliListOfDict(), indent=1)
        raw = '\n'.join([
            r'<%def name="body()">',
            r'${head_msg}',
            r'<pre style="text-align: left;">',
            r'${dump}',
            r'</pre>',
            r'<hr>',
            r'<%include file="article.html"/>',
            r'</%def>',
            r'',
            r'<%inherit file="basic.html"/>',
        ]).strip()
        return MakoRender(raw, locals().copy())
    # if settings.DEBUG:
    #     return HttpResponse(_GetPageDump().encode('utf-8'))
        
    def _GetPage():
        title = "Admin Page"
        raw = '\n'.join([
            r'<%def name="body()">',
            r'<%include file="article.html"/>',
            r'</%def>',
            r'',
            r'<%inherit file="basic.html"/>',
        ]).strip()
        return MakoRender(raw, locals().copy())
    return HttpResponse(_GetPage().encode('utf-8'))

def xdmin(request):
    print(json.dumps(Request2Dict(request)))
    def _GetPage():
        table_lol = [['x%d=%d'%(i, i**j) for j in range(3)] for i in range(10)]
        csrftoken = str(csrf(request)["csrf_token"])
        title = "XDmin"
        raw = '\n'.join([
            r'<%def name="body()">',
            r'<%include file="xdmin.html"/>',
            # r'${table_lol}'
            r'</%def>',
            r'',
            r'<%inherit file="basic.html"/>',
        ]).strip()
        return MakoRender(raw, locals().copy())
    if AlreadyLogin(request) and request.session['already_login_username']=='xdd':
        return HttpResponse(_GetPage().encode('utf-8'))
    raise Http404


def getPage(request, tag=None):
    csrftoken = str(csrf(request)["csrf_token"])
    title = "CNLogin"
    content = "<h1>⛱嗨</h1>"
    if tag=='dump':
        dump_content = json.dumps({
            'POST': request.POST,
            'GET': request.GET,
            'COOKIES': request.COOKIES,
            'session': dict(request.session),
        }, indent=1)
        
    raw = '\n'.join([
        r'<%def name="dump()">',
        r'<pre style="text-align: left;">',
        r'${dump_content}',
        r'</pre>',
        r'</%def>',
        r'',
        r'<%def name="body()">',
        r'${content}',
        r'% if tag=="dump":',
        r'${self.dump()}',
        r'% else:',
        r'<%include file="login.html"/>',
        r'<%include file="article.html"/>',
        r'% endif',
        r'</%def>',
        r'',
        r'<%inherit file="basic.html"/>',
    ]).strip()
    context = locals().copy()
    s0 = MakoRender(raw, context)
    return s0.encode('utf-8')