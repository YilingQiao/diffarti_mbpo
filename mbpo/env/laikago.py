import numpy as np
from gym import utils, spaces
import gym
import torch
import pytinydiffsim as pd
import api_diff

def calc_jac(y, x, free=False):
    ans = []
    py = y.reshape([-1])
    for i in range(py.shape[0]):
        flag = True
        if i == py.shape[0]-1 and free:
            flag = False
        ans.append(torch.autograd.grad(py[i], x, retain_graph=flag)[0])
    ans = torch.stack(ans, dim=0)
    return ans.reshape(y.shape+x.shape)

class LaikagoEnv(gym.Env, utils.EzPickle):
    """
        External forces (sim.data.cfrc_ext) are removed from the observation.
        Otherwise identical to Ant-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
    """
    def __init__(self, enable_qlimit=True, need_jac=True, mode=2, clip=10, random_init=1):
        utils.EzPickle.__init__(self)
        world = pd.TinyWorld(do_vis=False)
        parser = pd.TinyUrdfParser()
        convert_tool = pd.UrdfToMultiBody2()

        plane_mb = world.create_multi_body()
        plane_mb.isFloating = False
        plane_urdf_structures = parser.load_urdf('/homes/williamljb/icml/diffarti/data/plane_implicit.urdf')
        # world.do_convert_visuals(plane_urdf_structures)
        convert_tool.convert2(plane_urdf_structures, world, plane_mb)

        mb = world.create_multi_body()
        mb.isFloating = True
        urdf_structures = parser.load_urdf('/homes/williamljb/icml/diffarti/data/laikago/laikago_toes_zup.urdf')
        # world.do_convert_visuals(urdf_structures)
        convert_tool.convert2(urdf_structures, world, mb)

        deg90 = np.pi / 2
        # init_q = torch.tensor([0.0,0.0,0.0,1.0, 0.0,0.0,0.6, 0,1,0,1,0,-1,0,-1], dtype=torch.float32)
        # init_q = torch.tensor([0.0,0.0,0.0,1.0, 0.0,0.0,0.75, 0,1,0,1,0,-1,0,-1], dtype=torch.float32)
        init_q = torch.tensor([0.0,0.0,0.0,1.0, 0.0,0.0,1.5, 0.2,0,-0.5,0.2,0,-0.5,0.2,0,-0.5,0.2,0,-0.5], dtype=torch.float32)
        q_lo = torch.tensor([-70,-70,-70,-70,-70,-70,-70,-70,-70,-70,-70,-70], dtype=torch.float32) / 180. * np.pi
        q_hi = torch.tensor([70,70,70,70,70,70,70,70,70,70,70,70], dtype=torch.float32) / 180. * np.pi
        # q_lo = torch.tensor([-1,30,-1,30,-1,-70,-1,-70], dtype=torch.float32) / 180. * np.pi
        # q_hi = torch.tensor([1,70,1,70,1,-30,1,-30], dtype=torch.float32) / 180. * np.pi
        init_qd = torch.zeros([12+6], dtype=torch.float32)

        grav = pd.TinyVector3(pd.Utils.zero(), pd.Utils.zero(), pd.Utils.scalar_from_double(-9.81))

        dt = 1./100*5
        finer = 5
        world.dt = pd.Utils.scalar_from_double(dt/finer)

        self.output_file = None
        self.world = world
        self.dt = dt
        self.mode = mode
        self.clip = clip
        self.finer = finer
        self.wdt = self.dt / self.finer
        self.obs_siz = 12+5+12+6 #+8*3 #8+7+8+6 #
        self.act_siz = 12
        self.frame = 0
        self.init_q = init_q
        self.init_qd = init_qd
        self.q_lo = q_lo
        self.q_hi = q_hi
        self.mult = 1
        self.power = 150. * self.mult
        self.limit = 150. * self.mult
        self.mb = mb
        self.plane_mb = plane_mb
        self.grav = grav
        self.enable_qlimit = enable_qlimit
        self.need_jac = need_jac
        self.q = self.init_q + 0
        self.qd = self.init_qd + 0
        self.act_lr = 1
        self.random_init = random_init
        world.adj_initialize(grav, 1000, self.act_siz)
        world.friction = pd.Utils.scalar_from_double(1.)
        for l in mb.links:
            l.damping = pd.Utils.scalar_from_double(1.)
        self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=[self.obs_siz])
        self.action_space = spaces.Box(low=-1, high=1, shape=[self.act_siz])

    def get_ext(self, q0, qd, tau):
        # return torch.zeros([8])
        q = q0[-12:] + qd[-12:] * self.wdt
        # print('extq: ',q0[-8:])
        # print('extq: ',qd[-8:])
        # print('extq: ',q)
        if self.enable_qlimit:
            # dist0 = torch.where(self.q_hi < self.q[-8:], -torch.ones([8]), torch.zeros([8]))
            # dist1 = torch.where(self.q_lo > self.q[-8:], torch.ones([8]), torch.zeros([8]))
            dist0 = torch.min(self.q_hi - q, torch.zeros([12]))
            dist1 = torch.max(self.q_lo - q, torch.zeros([12]))
            force = torch.min(torch.max(10 * (dist0 + dist1), torch.tensor(-5)), torch.tensor(5))
            # force = 10 * (dist0 + dist1)
            # print('force: ',force)
            force = force * self.power / self.mult
            return tau + force #torch.where((dist0 + dist1) != 0, force, tau)
        return tau


    def RK4(self, q, qd, tau, world):
        t1 = self.get_ext(q, qd, tau)
        r10, r11 = api_diff.sim_layer(q, qd, t1, world)

        q10=((q+r10)/2); q11=((qd+r11)/2)
        t2 = self.get_ext(q10, q11, tau)
        r20, r21 = api_diff.sim_layer(q10, q11, t2, world)

        q20=((q+r20)/2); q21=((qd+r21)/2)
        t3 = self.get_ext(q20, q21, tau)
        r30, r31 = api_diff.sim_layer(q20, q21, t3, world)

        q30=r30; q31 = r31
        t4 = self.get_ext(q30, q31, tau)
        r40, r41 = api_diff.sim_layer(q30, q31, t4, world)

        return (r20+r30+r40+3*q)/6, (r21+r31+r41+3*qd)/6


    def is_done(self):
        return (self.q.detach().numpy())[6] < 0.4 or (self.q.detach().numpy())[6] > 2.0 or torch.isnan(self.q).any() or not self.success

    def get_state(self):
        self.state = torch.cat([
            self.q[:4],
            self.q[6:],
            self.qd*0.05,
            # joints
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ], dim=0)
        # print('obs: ',self.state)
        absstate = self.state.abs().detach().numpy()
        return self.state, [absstate[:4],absstate[4:13],absstate[13:]]#,absstate[27:]]

    def get_reward(self, a):
        yposafter = api_diff.get_com(self.q, self.world)[4]
        forward_reward = (yposafter - self.yposbefore)/self.dt
        self.yposbefore = yposafter.detach()
        ctrl_cost = .5 * (a**2).sum()
        survive_reward = 1.0
        self.reward = forward_reward - ctrl_cost + survive_reward
        if torch.isnan(self.reward):
            self.reward *= 0
            self.success = False
        if self.output_file is not None:
            print('r: ',self.reward, self.q[4])
        return self.reward, [forward_reward.detach().numpy(), ctrl_cost.detach().numpy()]

    def step(self, action):
        # import time
        # s=time.time()
        # print('[ ENV ]: action: ',action)
        if self.output_file is not None:
            for i in range(len(self.vert)):
                self.print_link(i)
        self.frame += 1
        self.action = action
        self.success = False

        # print('[ ENV ] step',self.frame)
        # print('q: ',self.q, self.qd)
        # print('a: ', action)
        # print('joint: ', api_diff.get_joints(self.q, self.world).reshape([-1,3])[[1,5,9,13,17]])
        # print(self.get_state()[0])
        # action = action * 0 #+ np.array([1,0,0,0,0,-0,0,-0])*1
        # action[1] = -1
        # if self.frame > 10:
        #     action[1] = -1
        # if self.frame == 900:
        #     while True:
        #         pass

        # tau = torch.tensor(action * self.power, dtype=torch.float32) #torch.clamp(action[0],-1,1) * self.power
        # tau = torch.clamp(tau, -self.limit, self.limit)
        tau = torch.tensor(action, dtype=torch.float32)
        tau.requires_grad = True
        # print('tau=',tau)
        # e=time.time();print('before sim:',e-s);s=e
        self.q = self.q.detach()
        self.qd = self.qd.detach()
        # try:
        for substep in range(self.finer):
            # self.q, self.qd = api_diff.sim_layer(self.q, self.qd, tau, self.world)
            # # print('\tq00: ',self.q[[6,8]],self.qd[[5,7]])
            # if self.enable_qlimit:
            #     clamped = torch.min(torch.max(self.q[-8:], self.q_lo), self.q_hi)
            #     diff = (clamped - self.q[-8:]) / (self.dt / self.finer)
            #     self.qd = torch.cat([self.qd[:-8], self.qd[-8:] + diff], axis=0)
            #     self.q = torch.cat([self.q[:-8], clamped], axis=0)
            # print('\tq01: ',self.q[[6,8]],self.qd[[5,7]])
            self.q, self.qd = self.RK4(self.q, self.qd, tau * self.power, self.world)
            # self.q, self.qd = api_diff.sim_layer(self.q, self.qd, tau + 1 * self.power * (dist0 + dist1), self.world)
            # print('\tq0: ',self.q[[6,7]],self.qd[6])
        self.success = True
        # except:
        #     pass
        # e=time.time();print('sim:',e-s);s=e
        # print('q=',self.q,self.qd)
        self.tau = tau
        # print('[ ENV ]: action done')
        _, state_dat = self.get_state()
        _, reward_dat = self.get_reward(tau)
        if self.need_jac:
            ind = np.random.randint(self.obs_siz)
            if self.mode == 0:
                mult = 0
            elif self.mode == 1:
                mult = 1
            else:
                mult = self.obs_siz
                if self.mode == 3:
                    self.act_lr = 1.0 / mult
            jac = calc_jac(self.state[ind], self.tau, free=False).reshape([1,-1]).numpy() * mult #* 0#self.obs_siz
            jacr = calc_jac(self.reward, self.tau, free=True).reshape([1,-1]).numpy()
            jac[np.isnan(jac)] = 0
            jacr[np.isnan(jacr)] = 0
            jac = np.clip(jac, -self.clip, self.clip) * self.act_lr
            jacr = np.clip(jacr, -self.clip, self.clip) * self.act_lr
        else:
            ind = 0
            jac = np.zeros([1, self.act_siz])
            # jac = np.zeros([self.obs_siz, self.act_siz])
            jacr = np.zeros([1,self.act_siz])
        return self.state.detach().numpy(), self.reward.detach().numpy(), self.is_done(), {
            'jac': jac, 'jacr': jacr, 'jacind': np.array([ind]), 'forward_reward': reward_dat[0], 'ctrl_cost': reward_dat[1],
            'state_rot': state_dat[0], 'state_pos': state_dat[1], 'state_vel': state_dat[2]}#, 'state_joint': state_dat[3]}

    def addvert(self, i):
        ans = []
        face = []
        with open('/homes/williamljb/icml/asset_ant/{}.obj'.format(i), 'r') as f:
            for lines in f:
                if lines[0] == 'v' and lines[1] == ' ':
                    ans.append(np.array([float(k) for k in lines[2:].split(' ')]))
                if lines[0] == 'f':
                    face.append([k.split('/')[0] for k in lines[2:].split(' ')])
        self.fcs.append(face)
        self.vert.append(ans)

    def print_link(self, i):
        with open(self.output_file + '/{}_{}.obj'.format(self.frame, i), 'w') as f:
            for vert in self.vert[i]:
                v = pd.TinyVector3(pd.Utils.scalar_from_double(vert[0]),
                        pd.Utils.scalar_from_double(vert[1]),
                        pd.Utils.scalar_from_double(vert[2]))
                world = self.mb.body_to_world(i-1, v)
                out = [pd.Utils.getDouble(world[0]),pd.Utils.getDouble(world[1]),pd.Utils.getDouble(world[2])]
                f.write('v {} {} {}\n'.format(out[0],out[1],out[2]))
            for face in self.fcs[i]:
                f.write('f {} {} {}\n'.format(face[0],face[1],face[2]))

    def reset(self):
        if self.output_file is not None:
            print('GOOD!')
            self.vert = []
            self.fcs = []
            for i in range(len(self.mb.links)+1):
                self.addvert(i)
        # print('reset!', self.q, self.qd)
        self.q = self.init_q + (torch.rand_like(self.init_q) * 0.2 - 0.1) * self.random_init
        self.qd = self.init_qd + (torch.randn_like(self.init_qd) * 0.1) * self.random_init
        # self.qd[7] = -1
        self.frame = 0
        self.reward = torch.tensor(0.)
        self.success = True
        self.yposbefore = api_diff.get_com(self.q, self.world)[4].detach()
        # self.q[7] = 0.5
        # joints = api_diff.get_joints(self.q, self.world)
        # print(self.q)
        # print(joints)
        # while True:
        #     continue
        return self.get_state()[0].numpy()

