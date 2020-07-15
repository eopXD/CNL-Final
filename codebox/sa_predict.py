import time, contextlib, requests
from codebox.sa import *
GPU, CPU = MakeTransportFunction()
CNL = lambda *args: os.path.join('/home/cnl', *args)
def retail():
    for i in range(1, 7):
        src = CNL('packet/packet'+str(i))
        dst = ('data/tail'+str(i)+'.csv')
        os.system(f"{CNL('tail_packet')} 3 {src} > {dst}")
        s0 = ''
        with open(dst, 'r') as f:
            s0 = f.readlines()
        with open(dst, 'w') as f:
            for i, s in enumerate(s0):
                if i<3:
                    continue
                print(s, file=f)
        print(f'retail {i} done')
        
def _script():
    parser = argparse.ArgumentParser(
        description="self attention position",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--model', default='', type=str, help='resume path to exist checkpoint file')
    add('--mac', default='', type=str, help='mac address to predict')
    add('--delay', default=0.5, type=float, help='polling')
    args = parser.parse_args()
    model_toload = args.model
    if type(model_toload)!=type('s0') or len(model_toload)<=0:
        model_toload = FindNewestFile('model') or ''
    
    with contextlib.redirect_stdout(sys.stderr):
        trn = Tester(
            {
                1: 'data/tail1.csv',
                2: 'data/tail2.csv',
                3: 'data/tail3.csv',
                4: 'data/tail4.csv',
                5: 'data/tail5.csv',
                6: 'data/tail6.csv',
            }, 
            args.mac,
            init_load=model_toload)
        print('Model Size = ', ModelSize(trn.func))
    
    st = now()
    for i in range(2147483647):
        with contextlib.redirect_stdout(sys.stderr):
            print(i, 'start', now()-st)
            retail()
        if len(trn.mac)>0:
            # spec
            x, y = trn.predict()
            print(x, y)
        else:
            # all
            print('--------==== all mac mode ====--------')
            sys.stderr.flush()
            trn.predict()
            print('XD')
    globals().update(locals())
    return 0
class Tester(Trainer):
    def __init__(self, fname_dict, mac, window_size=8, test_size=128, init_load=''):
        self.func = Func()
        if len(init_load)>0:
            self("hard load path", init_load)
            self.load(init_load)
        self.fname_dict = fname_dict
        self.window_size = window_size
        self.mac = mac
    def predict(self):
        self.prs = PacketFromSource(self.fname_dict)
        if len(self.mac)>0:
            return self.predict_mac(self.mac)
        mac_list = UniqueList([x[5] for x in self.prs])
        print('--------==== mac_list ====--------')
        print(mac_list)
        for mac in mac_list:
            try:
                print('trying predict mac =', mac)
                x, y = self.predict_mac(mac)
                print('x', x, 'y', y)
                url = f'https://linux7.csie.org:9090/pos?mac={mac}&x={x}&y={y}&tag=XD'
                print('url', url)
                requests.get(url, verify=False)
            except:
                print('skip mac', mac)
    def predict_mac(self, mac):
        prswin = PacketWindow(self.prs, window_size=self.window_size, target_mac=self.mac, skip=0)
        self.DL_test = DataLoader(prswin, 32, collate_fn=self.cfn)
        print('--------==== print(len(self.DL_test)) ====--------')
        print(len(self.DL_test))
        xlist, ylist = [], []
        for i_mini, (source_id, is_target, fet, _, _) in enumerate(self.DL_test):
            with torch.no_grad():
                ox, oy = self.func(source_id, is_target, fet)
            return float(torch.mean(ox)), float(torch.mean(oy))


# ================================================================
if __name__ == '__main__':
    sys.exit(_script())