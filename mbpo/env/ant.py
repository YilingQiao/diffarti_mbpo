import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

class AntTruncatedObsEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    """
        External forces (sim.data.cfrc_ext) are removed from the observation.
        Otherwise identical to Ant-v2 from
        https://github.com/openai/gym/blob/master/gym/envs/mujoco/ant.py
    """
    def __init__(self, skip=5, enable_contact=1):
        self.frame=0
        self.enable_contact = enable_contact
        self.output_file = None
        self.output = []
        mujoco_env.MujocoEnv.__init__(self, './ant.xml', skip)
        utils.EzPickle.__init__(self)

    def step(self, a):
        self.frame += 1
        # a = a*0+np.array([1,0,0,0,0,-0,0,-0])*1 # 4123
        # print('--------------------------')
        # print('frame: ',self.frame)
        # print('qs: ',self.sim.data.qpos, self.sim.data.qvel)
        # print('dat: ',self.frame_skip, self.dt, self.sim.data.time)
        # print(self.sim.data.qM)
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        # print(xposbefore,xposafter)
        # print('qs after: ',self.sim.data.qpos, self.sim.data.qvel)
        if self.output_file is not None:
            print('a: ',a)
        forward_reward = (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        # print('cost: ',ctrl_cost)
        # if self.frame == 11:
        #     while True:
        #         continue
        contact_cost = self.enable_contact*(0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1))))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        # if done:
        #     print('frame: ',self.frame)
        #     print('qs: ',self.sim.data.qpos, self.sim.data.qvel)
        #     while True:
        #         continue
        if self.output_file is not None:
            tmp = self.sim.data.qpos + 0
            self.output.append(tmp)
            print(self.frame, reward)
            
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        qvel = self.sim.data.qvel * 0.05
        if self.output_file is not None:
            print('obs:',np.concatenate([
                self.sim.data.qpos.flat[4:7],
                self.sim.data.qpos.flat[3:4],
                self.sim.data.qpos.flat[2:3],
                self.sim.data.qpos.flat[-2:],
                self.sim.data.qpos.flat[7:-2],
                qvel.flat[3:6],
                qvel.flat[:3],
                qvel.flat[-2:],
                qvel.flat[6:-2],
                # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
            ]))
        return np.concatenate([
            self.sim.data.qpos.flat[4:7],
            self.sim.data.qpos.flat[3:4],
            self.sim.data.qpos.flat[2:3],
            self.sim.data.qpos.flat[-2:],
            self.sim.data.qpos.flat[7:-2],
            qvel.flat[3:6],
            qvel.flat[:3],
            qvel.flat[-2:],
            qvel.flat[6:-2],
            # np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        self.frame=0
        qpos = self.init_qpos + 0
        # qpos[-8:] = [0,1,0,-1,0,-1,0,1] # 1234
        qpos = qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        # qpos[:3]=[0,0,0.6]
        # qpos[-8:] = [4,1,0,-1,0,-1,0,1] # 1234
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        # qvel[13] = -1
        self.set_state(qpos, qvel)
        if self.output_file is not None:
            if len(self.output) > 0:
                self.output = np.array(self.output)
                print(self.output)
                print('size=',self.output.shape)
                np.save(self.output_file, self.output)
            self.output = [qpos]
        # self.model.opt.gravity[-1] = -0
        # self.model.opt.gravity[0] = 0
        # self.model.opt.gravity[1] = 0
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5