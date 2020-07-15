import os, sys, json, argparse
from newremote.ptmux import *
class BasicCall(dict):
    def __init__(self, args):
        self.update(args.__dict__)
        self.run()
    def run(self):
        raise NotImplementedError()
    def __call__(self, *args):
        print(f'[{self.__class__.__name__}]', *args, file=sys.stderr)
    def shell(self, cmd, **kwargs):
        if 'buffer' in kwargs:
            self.setdefault('buffer', '')
            self['buffer'] += f'{cmd} ; '
            return 0
        if 'flush' in kwargs:
            self.shell(cmd, buffer=True)
            rc = self.shell(self['buffer'])
            del self['buffer']
            return rc
        self(cmd)
        return os.system(cmd)
class TmuxCall(BasicCall):
    def run(self):
        cmd = self['cmd']
        session = self['session']
        window = self['window']
        kill = self['kill'] # TODO: still got killed after rename
        overset = self['overset']
        if overset:
            self.shell(r'tmux set-option -g default-shell /bin/zsh')
            self.shell(r'tmux set-option -g remain-on-exit off')
            self.shell(r'tmux set -g base-index 1')
            self.shell(r'tmux setw -g pane-base-index 1')
            self.shell(r'tmux bind m set -g mouse on')
            self.shell(r'tmux bind M set -g mouse off')
            self.shell(r'tmux set-window-option -g allow-rename off')
            self.shell(r'tmux set-window-option -g automatic-rename off')
        self(f'(args kill = {kill})')
        if kill:
            self.shell(f'tmux kill-window -t {session}:{window}')
        twin = TmuxWindow(Tmux(), session, window, kill)
        twin(cmd)
        # self.shell(f'tmux new-session -A -s {session} -n sleep60 "sleep 60"')
        # self.shell(f'tmux new-window -k -n {window}')
        # self.shell(f'tmux send-keys '
        #     f'-t {session}:{window} '
        #     f'"{cmd}" ENTER')
    
# ================================================================
def _script():
    parser = argparse.ArgumentParser(
        description="tmux attacher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add = parser.add_argument
    add('cmd', nargs='?', default="echo $(hostname)", type=str, 
        help="command to run (keyboard simulated)")
    add('--session', default="agent", type=str, 
        help="session name")
    add('--window', default="nowindow", type=str, 
        help="window name")
    add('--kill', default=False, action='store_true', 
        help='kill window if exists (use it carefully!)')
        # help='kill ALL window under the session (use it carefully!)')
    add('--overset', default=False, action='store_true', 
        help='overwrite settings before make session')
    args = parser.parse_args()
    TmuxCall(args)

if __name__ == '__main__':
    _script()
