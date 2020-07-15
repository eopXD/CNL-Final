# utility library
# DO NOT import custom libraries
import os, sys, json, inspect
import math
import datetime
import hashlib
from typing import Callable, List, Dict, Optional, ClassVar
from collections import OrderedDict, defaultdict

import socket
from contextlib import closing, suppress
import tempfile
import multiprocessing as mp

import numpy as np
from numpy.random import RandomState
def FindNewestFile(dirname: str, hour=24, min_byte=32, more=False):
    r'''return newest file path in the directory'''
    x = os.listdir(dirname)
    # print('loading dir', dirname)
    # print(x)
    x = [{
        'fname': f,
        'path': os.path.realpath(os.path.join(dirname, f)),
    } for f in x]
    lod = []
    for d in x:
        ts = os.stat(d['path']).st_mtime
        t = datetime.datetime.fromtimestamp(ts)
        d['time'] = t.strftime("%m/%d %H:%M:%S")
        d['size'] = os.stat(d['path']).st_size
        d['ts'] = ts
        if d['size']>=min_byte and now()-ts<=hour*3600:
            lod.append(d)
    if len(lod)==0:
        return (None, None) if more else None
    lod = sorted(lod, key=lambda d:-d['ts'])
    if more:
        return lod[0]['path'], lod
    return lod[0]['path']
class CallableDict(dict):
    r'''base class kind of manager, easy to log'''
    def __init__(self, fn=None, dict_init=None):
        self._fn = fn or self.get_default_fn()
        if dict_init:
            self.update(dict_init)
    def __call__(self, *args):
        return self._fn(*args)
    def get_default_fn(self):
        fn = lambda *args: print(*args, file=sys.stderr)
        return fn


from io import StringIO 
def print2str(*args, **kwargs):
    buf = StringIO()
    print(*args, **kwargs, file=buf)
    return buf.getvalue()
def qnt(numpy_like, grid: int=21):
    r'''quantile for short'''
    return np.percentile(numpy_like, FSpace(0, 100, grid))

def FlattenList(lst: list)->list:
    rst = []
    for element in lst:
        if type(element)==type(rst):
            rst.extend(FlattenList(element))
        else:
            rst.append(element)
    return rst

def SplitLength(total, ratio_list: list)->list:
    len_list = [max(1, int(math.ceil(total*ratio))) for ratio in ratio_list]
    if sum(len_list) != total:
        delta = total - sum(len_list)
        idx = int(np.argmax(len_list))
        len_list[idx] += delta
        if len_list[idx]<1:
            raise RuntimeError("SplitLength skew ratio?")
    return len_list
class RunningStat:
    def __init__(self, m=0.1, record_series=True):
        self.m = m
        self.num = 0
        self.mean = 0
        self.std = 0
        self.max = None
        self.min = None
        self.record_series = record_series
        self.series = []
    def step(self, new_sample):
        r'''
        add new_sample, can be float/int/list/np.array
        if only one element its std will infer from running mean
        '''
        try:
            assert len(new_sample)>=1
        except TypeError:
            new_sample = [new_sample]
        if self.record_series:
            self.series.extend(new_sample)
        
        m = self.m
        new_mean, new_std, new_max, new_min = (
            np.mean(new_sample), np.std(new_sample), 
            np.max(new_sample), np.min(new_sample) )
        if self.num==0:
            m = 1
            self.max = new_max
            self.min = new_min
        self.mean = (1-m) * self.mean + m * new_mean
        if len(new_sample)==1:
            new_std = new_sample[0] - self.mean # kinda MAE
        self.std  = (1-m) * self.std  + m * new_std
        self.max = max(self.max, new_max)
        self.min = min(self.min, new_min)
        self.num += len(new_sample)
        return self.mean, self.std
    def __str__(self):
        return "%9.2f(%9.2f)"%(self.mean, self.std)
    def __repr__(self):
        return self.__str__()

def SharedDict():
    r'''Get multiprocessing shared object'''
    return mp.Manager().dict()
class ForkHelper:
    r'''
    support simple "fork all and wait all" procedure
    [:Example:]
    shared = SharedDict()
    forker = ForkHelper()
    for idx in range(50):
        if forker.fork_parent():
            continue
        try:
            # do something
            shared[idx] = result
        except BaseException as e:
            # record error
            shared[idx] = str(e)
        finally:
            os._exit(0)
    forker.wait()
    '''
    def __init__(self, verbose=False):
        self.pid_list = []
        self.verbose = verbose
    def __len__(self):
        return len(self.pid_list)
    def __enter__(self):
        return self
    def __exit__(self, exit_type, exit_value, traceback):
        self.wait()
    def fork_parent(self)->bool:
        r'''fork and tells if it is a parent process'''
        pid = os.fork()
        if pid==0:
            return False
        self.pid_list.append(pid)
        if self.verbose:
            print("[ForkHelper] fork pid %d"%pid, file=sys.stderr)
        return True
    def wait(self):
        r'''blocking wait all pid
        (make sure no deadlock in your children!)'''
        for i, pid in enumerate(self.pid_list):
            with suppress(ChildProcessError):
                os.waitpid(pid, 0)
            if self.verbose:
                print("[ForkHelper] wait pid %d (%3d/%3d)"%(pid, i+1, len(self)), file=sys.stderr)
def ForkMany(work_fn: Callable, shared, arg_list, verbose=False):
    with ForkHelper(verbose=verbose) as forker:
        for arg in arg_list:
            if forker.fork_parent():
                continue
            try:
                shared[arg] = work_fn(arg)
            except BaseException as e:
                if verbose:
                    print("[ForkMany] exception during work_fn\n%s"%str(e), file=sys.stderr)
                    sys.stderr.flush()
                shared[arg] = str(e)
            finally:
                os._exit(0)

def MakeTemp(s0=None):
    f = tempfile.NamedTemporaryFile(mode="w")
    if type(s0)!=type(None):
        f.file.write(s0)
        f.file.flush()
    return f # dont close if you need the file!

# source:
# https://stackoverflow.com/a/35370008
def checkSocket(host, port):
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        return sock.connect_ex((host, port)) == 0

def escapeShell(x: str) -> str:
    return x.replace("\\", "\\\\").replace("\"", "\\\"").replace("$", "\\$")

def jprint(jlike_dict, indent: int=1, uprint: Callable=print):
    """
    Print list/dict/string in a json readable way
    """
    uprint(json.dumps(jlike_dict, indent=indent, default=lambda x: "JSON_FAIL"))


"""
Get timestamp in seconds float
A convenient alias :p
"""
now = lambda: datetime.datetime.now().timestamp()

class RuleDict:
    r'''cleaner version of ChainDict'''
    def __init__(self, target: Dict):
        self.target = target
        self.rule_table = {}
    def add_rule(self, short_key: str, *long_key):
        if short_key not in self.rule_table:
            self.rule_table[short_key] = []
        self.rule_table[short_key].append(long_key)
    def get_dict(self):
        return {key: self[key] for key in self.rule_table.keys()}
    def update(self, dsource: Dict):
        for k, v in dsource.items():
            self.__setitem__(k, v)
    def update_soft(self, dsource: Dict):
        intkey = set.intersection(set(self.rule_table.keys()), set(dsource.keys()))
        for key in intkey:
            self.__setitem__(key, dsource[key])
    def __getitem__(self, short_key):
        try:
            # always read from first rule one
            long_key = self.rule_table[short_key][0]
            result = self.target
            for key in long_key:
                result = result[key]
            return result
        except KeyError:
            return None
    def __setitem__(self, short_key, val):
        for long_key in self.rule_table[short_key]:
            target = self.target
            for key in long_key[:-1]:
                target = target[key]
            target[long_key[-1]] = val
    


# --------==== Machine Learning ====--------
def GetShuffledIdxList(len_of_list: int, seed: int=1337) -> List:
    idx_list = [i for i in range(len_of_list)]
    random1337 = RandomState(seed=seed)
    random1337.shuffle(idx_list)
    return idx_list
def GetChunk(lst: List, chunk_size: int)->List:
    r'''
    seperate lst per chunk_size, for example:
    GetChunk([0, 1, 2, 3, 4], 2)
    -> [[0, 1], [2, 3], [4]]
    '''
    lol = []
    off = 0
    while True:
        r = lst[off:off+chunk_size]
        if len(r)==0:
            break
        lol.append(r)
        off += chunk_size
    return lol
def ijoin(string_joinable: str, object_list: list, convert: Callable=lambda x: str(x))->str:
    r'''fast generate "3 1 7" by ijoin(' ', [3, 1, 7])'''
    return string_joinable.join([convert(x) for x in object_list])
def GetSubsetList(suplist: List, idx_in_use: set) -> List:
    suplist_out = []
    for idx in range(len(suplist)):
        if idx in idx_in_use:
            suplist_out.append(suplist[idx])
    return suplist_out
def UniqueList(lst: List)->List:
    return list(OrderedDict.fromkeys(lst))
def SoftUpdate(dtarget: Dict, dsource: Dict):
    intkey = set.intersection(set(dtarget.keys()), set(dsource.keys()))
    for key in intkey:
        dtarget[key] = dsource[key]

def FRange(st, ed, step, rod=4, endpoint=False):
    if endpoint:
        ed = ed+0.000977 # eps of float16
    return [round(float(x), rod) for x in np.arange(st, ed, step)]
def FSpace(st, ed, num, rod=4, endpoint=True):
    resultNP = np.linspace(st, ed, num, endpoint=endpoint)
    return [round(float(x), rod) for x in resultNP]
def LFSpace(st, ed, num, rod=4, endpoint=True):
    assert st>0 and ed>0
    resultF = FSpace(math.log(st), math.log(ed), num, rod, endpoint)
    return [round(float(math.exp(x)), rod) for x in resultF]

def IRange(st, ed, step, rod=0, endpoint=False):
    """
    Call np.arrange and convert it for you
    rod=-2 makes a 1337 into 1300
    """
    lst = [int(x) for x in np.arange(st, ed, step)]
    if endpoint:
        lst.append(int(ed))
    lst = UniqueList([int(round(int(x), rod)) for x in lst])
    return lst
def ISpace(st, ed, num, rod=0, endpoint=True):
    resultF = FSpace(float(st), float(ed), num, 9999, endpoint)
    if endpoint:
        resultF[-1] = ed
    return UniqueList([int(round(int(x), rod)) for x in resultF])
def LISpace(st, ed, num, rod=0, endpoint=True):
    resultF = LFSpace(float(st), float(ed), num, 9999, endpoint)
    if endpoint:
        resultF[-1] = ed
    return UniqueList([int(round(int(x), rod)) for x in resultF])


# --------==== File System ====--------
def md5(x: str) -> str:
    return hashlib.md5(x.encode('utf-8')).hexdigest()

def _getFileMTime(fpath: str) -> float:
    try:
        t = os.path.getmtime(fpath)
    except FileNotFoundError: # no such file
        t = -1
    return t
    
def isUpToDate(target_path: str, *args) -> bool:
    t_newest = 0
    t1 = _getFileMTime(target_path)
    for fpath_i in args:
        assert type(fpath_i) == type("")
        ti = _getFileMTime(fpath_i)
        if t_newest < ti:
            t_newest = ti
    return t1 > t_newest

def GetJson(apath: str):
    """
    Load specific json file into python object in one line.
    Any error raised will lead to "None" result.
    """
    try:
        with open(apath, "r") as f:
            file_str = f.read().strip()
            if len(file_str)==0:
                return None
            return json.loads(file_str)
    except FileNotFoundError:
        return None
def GetJsonFP(fp):
    try:
        # s = fp.read()
        return json.load(fp)
    except FileNotFoundError:
        return None
def SetJsonSoft(apath: str, jobj)->bool:
    """
    Check if file contents are the same before calling `json.dumps`
    It preserves metadata "last-modified", which is useful to GNUMake
    """
    s1 = json.dumps(jobj, indent=1, sort_keys=True)
    try:
        with open(apath, 'r') as f:
            s0 = f.read()
        if s0==s1:
            return False
    except FileNotFoundError:
        pass
    with open(apath, 'w') as f:
        f.write(s1)
    return True
def ReadFile(apath: str)->str:
    with open(apath, "r") as f:
        return f.read()
def Write(apath: str, str_like)->bool:
    try:
        with open(apath, "w") as f:
            f.write(str_like)
        return True
    except FileNotFoundError:
        return False

def WriteSoft(apath: str, str_like)->bool:
    try:
        with open(apath, 'r') as f:
            s0 = f.read()
        if s0==str_like:
            return False
    except FileNotFoundError:
        pass
    try:
        with open(apath, "w") as f:
            f.write(str_like)
        return True
    except FileNotFoundError:
        return False

# --------==== psutil ====--------
try:
    NO_PSUTIL = False
    import psutil
    def KillIPython(kill_self = False):
        last_process = None
        for p in psutil.process_iter(attrs=['name']):
            if 'ipykernel_launcher' in p.cmdline():
                if p.pid != os.getpid():
                    print("kill %s"%p)
                    p.kill()
                else:
                    last_process = p
        if type(last_process) != type(None):
            if kill_self:
                print("kill %s [kill self]"%last_process)
                last_process.kill()
except ImportError:
    NO_PSUTIL = True
# --------==== Pandas ====--------
try:
    NO_PANDAS = False
    import pandas as pd
    def DF(lod: List[Dict], col: Optional[List]=None):
        if type(col)==type(None):
            return pd.DataFrame(lod)
        return pd.DataFrame(lod, columns=col)
    def DFO(lod):
        col = list(lod[-1].keys())
        return pd.DataFrame(lod, columns=col)
except ImportError:
    NO_PANDAS = True

# --------==== Mako Template System ====--------
try:
    NO_MAKO = False
    from mako.template import Template
    from mako.lookup import TemplateLookup
    def MakoExtraOption()->dict:
        return {
            'input_encoding': 'utf-8',
            'output_encoding': 'utf-8',
            'default_filters': ['decode.utf8'],
            'encoding_errors': 'replace',
            'cache_enabled': 'False',
        }
    def MakoContextFilter(template_var_in: Optional[Dict]=None):
        template_var_in = template_var_in or {}
        template_var = template_var_in.copy()
        taboo_list = ['self', 'context']
        for taboo in taboo_list:
            if taboo in template_var:
                del template_var[taboo]
        return template_var
    def MakoRender(template_string: str, template_var_in: dict, lookup_dir: list=['html']):
        lookup = TemplateLookup(directories=lookup_dir, **MakoExtraOption())
        template_var = MakoContextFilter(template_var_in)
        return Template(
            template_string.strip(), 
            lookup=lookup, **MakoExtraOption()
        ).render_unicode(**template_var)
    def MakoRenderFile(template_path: str, template_var_in: dict, lookup_dir: list=['html']):
        lookup = TemplateLookup(directories=lookup_dir, **MakoExtraOption())
        template_var = MakoContextFilter(template_var_in)
        return Template(
            filename=template_path,
            lookup=lookup, **MakoExtraOption()
        ).render_unicode(**template_var)
    class MakoManager:
        r'''A configurable mako wrapper'''
        def __init__(self, lookup_main_dir: str):
            self.extra_option = MakoExtraOption()
            self.lookup = TemplateLookup(directories=[lookup_main_dir], **self.extra_option)
        def render(self, tmpl:str, ctx: Optional[Dict]=None):
            r'''render raw string ``tmpl`` in mako'''
            return Template(tmpl,
                lookup=self.lookup, **self.extra_option
                ).render_unicode(**MakoContextFilter(ctx))
        def render_file(self, tmpl_file:str, ctx: Optional[Dict]=None):
            r'''render plain file ``tmpl_file`` in mako
            (notice that there is no lookup)'''
            return Template(filename=tmpl_file,
                lookup=self.lookup, **self.extra_option
                ).render_unicode(**MakoContextFilter(ctx))
except ImportError:
    NO_MAKO = True