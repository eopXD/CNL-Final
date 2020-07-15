import argparse, pickle
from codebox.torchbase import *
GPU, CPU = MakeTransportFunction()
# max_csv = 512
max_csv = 65536
# max_csv = 2147483647
def _script():
    parser = argparse.ArgumentParser(
        description="self attention position",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('--train', default='', type=str, help='resume path to exist checkpoint file')
    add('--predict', default='', type=str, help='resume path to exist checkpoint file')
    add('--mac', default='', type=str, help='mac address to predict')
    args = parser.parse_args()
    if len(args.predict)>0:
        return _script_predict(args.predict, args.mac)
    
    
    model_toload = args.train
    if type(model_toload)!=type('s0') or len(model_toload)<=0:
        model_toload = FindNewestFile('model') or ''
    
    trn = Trainer({
        1: 'data/labeled_packet1.csv',
        2: 'data/labeled_packet2.csv',
        3: 'data/labeled_packet3.csv',
        4: 'data/labeled_packet4.csv',
        5: 'data/labeled_packet5.csv',
        6: 'data/labeled_packet6.csv', }, init_load=model_toload)
    print('Model Size = ', ModelSize(trn.func))
    
        
        
    st = now()
    for i in range(2147483647):
        print(i, 'start', now()-st)
        trn.run_epoch()
        trn.save(f'model/sa8.{i}')
    globals().update(locals())
    return 0

def _script_predict(cpath, mac):
    pass

class Func(nn.Module):
    def __init__(self, fet_dim=2, emb_dim=8):
        r'''
        input: (B, Features, PacketThroughTime)
        output: x=(B, 1), y=(B, 1)
        '''
        super().__init__()
        in_dim = fet_dim+emb_dim*2
        self.emb_source = nn.Embedding(10, emb_dim)
        self.emb_is_mac = nn.Embedding(2, emb_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.attn1 = Attention1d(in_dim, 8, 8)
        self.flat = nn.AdaptiveAvgPool1d(8)
        self.fc1 = nn.Linear(8*8, 51)
        self.fc2 = nn.Linear(8*8, 51)
        self.val = QuantileRegression(51)
    def forward(self, src, ism, x):
        B = x.size(0)
        xs = self.emb_source(src).transpose(1, 2)
        xi = self.emb_is_mac(ism).transpose(1, 2)
        x = torch.cat([x, xs, xi], dim=1)
        x = self.bn1(x)
        x = self.attn1(x)
        x = self.flat(x)
        x = x.view(B, -1)
        outx = self.val(self.fc1(x))
        outy = self.val(self.fc2(x))
        return outx, outy

GPU, CPU = MakeTransportFunction()
class Trainer:
    def __init__(self, fname_dict, window_size=32, test_size=128, init_load=''):
        self.func = Func()
        if len(init_load)>0:
            self("hard load path", init_load)
            self.load(init_load)
        prs = PacketFromSource(fname_dict)
        mac_list = ['20395658eeb8',
                    '38539cc0ef2f',
                    '98e79a5d012f',
                    'd4a33dcff535',
                    'fc183c5d8588']
        print('prepare', mac_list[0])
        prswin = PacketWindow(prs, window_size=window_size, target_mac=mac_list[0], skip=window_size//2)
        for i in range(1, len(mac_list)):
            print('prepare', mac_list[i])
            prswin += PacketWindow(prs, window_size=window_size, target_mac=mac_list[i], skip=window_size//2)
        prswin_train, prswin_test = random_split(prswin, [len(prswin)-test_size, test_size])
        self.prswin_train = prswin_train
        self.prswin_test = prswin_test
        print(len(prs), len(prswin), len(prswin_train), len(prswin_test))
        self.DL = DataLoader(prswin_train, 32, shuffle=True, drop_last=True, collate_fn=self.cfn)
        self.DL_test = DataLoader(prswin_test, 32, collate_fn=self.cfn)
        self.runloss_train = RunningStat()
        self.runloss_test = RunningStat()
        self.optimizer = torch.optim.Adam(self.func.parameters(), lr=0.01)
    @staticmethod
    def cfn(batch):
        batch = default_collate(batch)
        batch[0] = torch.stack(batch[0], dim=-1)
        batch[1] = torch.stack(batch[1], dim=-1)
        batch[2] = torch.stack(batch[2], dim=-1).float()
        batch[3] = torch.stack(batch[3], dim=-1).float()
    #     batch[4] = torch.stack(batch[4], dim=-1)
    #     batch[5] = torch.stack(batch[5], dim=-1)
        return (
            itns(batch[0]),
            itns(batch[1]),
            ftns(torch.stack([batch[2], batch[3]], dim=1)),
            batch[4].float(), # ansx
            batch[5].float(), # ansy
        )
    
    def run_epoch(self):
        Loss = nn.MSELoss()
        for i_mini, (source_id, is_target, fet, ansx, ansy) in enumerate(self.DL):
            self.optimizer.zero_grad()
            ox, oy = self.func(source_id, is_target, fet)
            loss = Loss(ox, ansx) + Loss(oy, ansy)
            loss.backward()
            self.optimizer.step()
            self.runloss_train.step(CPU(loss))
            should_verbose = (i_mini % 50 == 0)
            if should_verbose:
                self.update_test_set()
                print(f'#{i_mini:4d}', 'train:', self.runloss_train.mean,'test:', self.runloss_test.mean)
    def update_test_set(self):
        Loss = nn.MSELoss()
        loss_test_list = []
        for t_mini, (source_id, is_target, fet, 
                    ansx, ansy) in enumerate(self.DL_test):
            with torch.no_grad():
                ox, oy = self.func(source_id, is_target, fet)
                loss_test = Loss(ox, ansx) + Loss(oy, ansy)
                loss_test_list.append(CPU(loss_test))
        self.runloss_test.step(loss_test)
    def __call__(self, *args):
        print(*args)
    def load(self, checkpoint_path: str):
        r'''
        load chkpt_dict. Notice it wont do GPU transport
        many exceptions here, so pay attention!
        '''
        assert os.access(checkpoint_path, os.R_OK), 'checkpoint not readable'
        checkpoint_path = os.path.realpath(checkpoint_path)
        with open(checkpoint_path, 'rb') as f:
            chkpt_dict = pickle.load(f)
        assert type(chkpt_dict)==type({}), 'init_load got non chkpt_dict'
        assert 'func' in chkpt_dict, 'init_load got no func in chkpt_dict'
        self._apply_chkpt(chkpt_dict)
        return checkpoint_path
    def soft_load(self, fname):
        if os.access(fname, os.R_OK):
            self(f'Soft load {fname}')
            try:
                self.load(fname)
                return fname
            except (RuntimeError, KeyError):
                self(f'load fail')
        else:
            self(f'Soft load failed, file [{fname}] not found')
        return None
    def _apply_chkpt(self, chkpt_dict: dict):
        r'''chkpt_dict -> attrs, for `load` use'''
        self.func.load_state_dict(chkpt_dict['func'])
    def save(self, checkpoint_path: str):
        r'''
        save chkpt_dict, if such path used, it append some `_` after path
        return realpath saved
        update the `model_out` attr in report
        '''
        while os.access(checkpoint_path, os.F_OK):
            checkpoint_path += '_'
        checkpoint_path = os.path.realpath(checkpoint_path)
        chkpt_dict = self._fetch_chkpt()
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(chkpt_dict, f)
        self(f'save {checkpoint_path}')
        return checkpoint_path
    def _fetch_chkpt(self):
        r'''attrs -> chkpt_dict, for `save` use'''
        return {
            'func': self.func.state_dict(),
        }


# ================================================================

class Attention1dCell(nn.Module):
    def __init__(self, dim):
        r'''
        functional module calculates attention
        says dim=2 below
        input:
        (B, *, Q, d) query
        (B, *, K, d) key
        (B, *, K, v) value
        output:
        (B, *, Q, v)
        '''
        super().__init__()
        self.dim = dim
    def forward(self, qry, key, val, reg=0):
        qdim = self.dim
        ddim = self.dim+1
        # dot_scale = key.size(ddim) ** -0.5
        dot_scale = 1
        key = key.transpose(qdim, ddim)
        att = torch.matmul(qry, key)
        if reg>0:
            att += reg * torch.eye(att.size(qdim), out=torch.empty_like(att))
        att = torch.softmax(att * dot_scale, dim=qdim)
        out = torch.matmul(att, val)
        return out

class Attention1d(nn.Module):
    def __init__(self, in_dim, d_dim, v_dim):
        r'''
        input: (B, in, *)
        output: (B, v, *)
        '''
        super().__init__()
        self.getq = nn.Conv1d(in_dim, d_dim, 3)
        self.getk = nn.Conv1d(in_dim, d_dim, 3)
        self.getv = nn.Conv1d(in_dim, v_dim, 3)
        self.attn = Attention1dCell(dim=1)
    def forward(self, x):
        q = self.getq(x).transpose(1, 2)
        k = self.getk(x).transpose(1, 2)
        v = self.getv(x).transpose(1, 2)
        return self.attn(q, k, v).transpose(1, 2)
# ================================================================
# data
class PacketRaw(Dataset):
    def __init__(self, fname='data/small3.csv', source_id=3):
        self.source_id = source_id
        self.df = pd.read_csv(fname, header=None, nrows=max_csv) # ts, db, x, y, mac
    def get(self, key, idx):
        if key=='source_id':
            return self.source_id
        key = {
            'timestamp': 0,
            'db': 1,
            'x': 2,
            'y': 3,
            'mac': 4,
        }[key]
        return self.df.iloc[idx, key]
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        return (
            int(self.get('source_id', idx)),
            float(self.get('timestamp', idx)),
            float(self.get('db', idx)),
            float(self.get('x', idx)),
            float(self.get('y', idx)),
            str(self.get('mac', idx)),
        )
class PacketFromSource(Dataset):
    r'''merge different PI packets and sort by time'''
    def __init__(self, fname_dict={
        3: 'data/small3.csv',
        4: 'data/small4.csv',
    }):
        self.lol = []
        for source_id, fname in fname_dict.items():
            pr = PacketRaw(fname, source_id)
            print('PacketFromSource', source_id, fname, 'len=', len(pr))
            for i in range(len(pr)):
                self.lol.append(pr[i])
        df = DF(self.lol).sort_values(1) # timestamp
        df[1] = df[1]*1e-18
        # df[2] = (df[2]-df[2].mean())/df[2].std()
        self.df = df
        print(df.tail())
    def __len__(self):
        return len(self.df)
    def get(self, keys, idx):
        keyi = {
            'source_id': 0,
            'timestamp': 1,
            'db': 2,
            'x': 3,
            'y': 4,
            'mac': 5,
        }[keys]
        return self.df.iloc[idx, keyi]
    def __getitem__(self, idx):
        return (
            int(self.get('source_id', idx)),
            float(self.get('timestamp', idx)),
            float(self.get('db', idx)),
            float(self.get('x', idx)),
            float(self.get('y', idx)),
            str(self.get('mac', idx)),
        )

class PacketWindow(Dataset):
    def __init__(self, source_ds, window_size=8, target_mac='38539cc0ef2f', skip=0):
        self.source_ds = source_ds
        self.window_size = window_size
        self.target_mac = target_mac
        self.ok_tail = []
        last_ok=-999
        for i in range(window_size-1, len(self.source_ds)):
            if last_ok+skip>=i:
                continue
            if self.source_ds[i][-1]==self.target_mac:
                self.ok_tail.append(i)
                last_ok = i
    def __len__(self):
        return len(self.ok_tail)
    def __getitem__(self, ok_idx):
        topack = {
            'source_id': [],
            'is_target': [],
            'timestamp': [],
            'db': [],
        }
        ansx = 0
        ansy = 0
        idx = self.ok_tail[ok_idx]
        for t in reversed(range(self.window_size)):
            idx_t = idx-t
            (source_id, timestamp, db, 
                x, y, mac) = self.source_ds[idx_t]
            topack['source_id'].append(source_id)
            topack['is_target'].append(int(mac==self.target_mac))
            topack['timestamp'].append(timestamp)
            topack['db'].append(db)
            ansx = x
            ansy = y
        return (
            topack['source_id'], 
            topack['is_target'], 
            topack['timestamp'], 
            topack['db'], 
            ansx,
            ansy,
        )
        # source_id, is_target, timestamp, db, ansx, ansy
# prs = PacketFromSource()
# prswin = PacketWindow(prs)
# print(len(prs), len(prswin))

# ================================================================
if __name__ == '__main__':
    sys.exit(_script())