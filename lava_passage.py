import itertools
import random

import numpy
from gym_minigrid.minigrid import *

class LavaPassageEnv(MiniGridEnv):
    """
    Narrow passage between lavas to goal
    """

    # Enumeration of possible actions
    class Actions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    # def __init__(self, is_lava = False, lava_once = True):
    def __init__(self, lava_reward=0):
        self.lava_steps = 0
        self.lava_reward = lava_reward
        self.episodes = 0
        super().__init__(height=9, width=9, max_steps=400,agent_view_size=5)
        self.action_space = spaces.Discrete(len(self.actions))
        self.actions = LavaPassageEnv.Actions
        self.action_space = spaces.Discrete(len(self.actions))



    # def _reward(self):
    #     return 1

    def gen_obs(self):
        """
        Generate one-hot encoding of position
        """

        one_hot_x = numpy.zeros(self.width)
        one_hot_y = numpy.zeros(self.height)
        one_hot_x[self.agent_pos[0]] = 1
        one_hot_y[self.agent_pos[1]] = 1
        cell = self.grid.get(self.agent_pos[0], self.agent_pos[1])

        obs = {
            'direction': self.agent_dir,
            'x': one_hot_x,
            'y': one_hot_y,
            'is_lava': [1] if cell and (cell.type == 'lava') else [0]
        }

        return obs

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        lava = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        lava = [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 1, 1, 0, 0, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 1, 1, 1, 1, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
        ]

        lava = np.array(lava)


        # print(lava)
        for x in range(lava.shape[1]):
            for y in range(lava.shape[0]):
                if lava[y, x]:
                    self.put_obj(Lava(), x, y)

        self.agent_dir = 0
        self.agent_pos = (1, 4)


        self.put_obj(Goal(), 7, 4)

    def step(self, action):
        self.step_count += 1

        reward = 0
        done = False
        stepped_into_lava = False;

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = self.grid.get(*fwd_pos)

        # Rotate left
        #print(action)
        if action == self.actions.left:
            self.agent_dir -= 1
            if self.agent_dir < 0:
                self.agent_dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.agent_dir = (self.agent_dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                self.agent_pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = 10
            if fwd_cell != None and fwd_cell.type == 'lava':
                self.lava_steps += 1
                reward = self.lava_reward
                stepped_into_lava = True

        # Pick up an object
        elif action == self.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if self.carrying is None:
                    self.carrying = fwd_cell
                    self.carrying.cur_pos = np.array([-1, -1])
                    self.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == self.actions.drop:
            if not fwd_cell and self.carrying:
                self.grid.set(*fwd_pos, self.carrying)
                self.carrying.cur_pos = fwd_pos
                self.carrying = None

        # Toggle/activate an object
        elif action == self.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(self, fwd_pos)

        # Done action (not used by default)
        elif action == self.actions.done:
            pass

        else:
            assert False, "unknown action"

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()
        if done:
            self.episodes += 1
        else:
            reward = -0.1
        return obs, reward, done, {'cost': 5. if stepped_into_lava else 0., 'begin': True if self.step_count == 1 else False}

    def reset(self):
        return super().reset()


if __name__ == "__main__":
    import argparse
    from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
    from gym_minigrid.window import Window


    def redraw(img):
        if not args.agent_view:
            img = env.render("rgb_array", tile_size=args.tile_size)

        window.show_img(img)


    def reset():
        if args.seed != -1:
            env.seed(args.seed)

        obs = env.reset()

        if hasattr(env, "mission"):
            print("Mission: %s" % env.mission)
            window.set_caption(env.mission)

        redraw(obs)


    def step(action):
        obs, reward, done, info = env.step(action)
        print("step=%s, reward=%.2f" % (env.step_count, reward))

        if done:
            print("done!")
            reset()
        else:
            redraw(obs)


    def key_handler(event):
        print("pressed", event.key)

        if event.key == "escape":
            window.close()
            return

        if event.key == "backspace":
            reset()
            return

        if event.key == "left":
            step(env.actions.left)
            return
        if event.key == "right":
            step(env.actions.right)
            return
        if event.key == "up":
            step(env.actions.forward)
            return

        if event.key == "enter":
            step(env.actions.done)
            return


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="random seed to generate the environment with",
        default=-1,
    )
    parser.add_argument(
        "--tile_size", type=int, help="size at which to render tiles", default=32
    )
    parser.add_argument(
        "--agent_view",
        default=False,
        help="draw the agent sees (partially observable view)",
        action="store_true",
    )

    args = parser.parse_args()

    env = LavaPassageEnv()
    obs = env.reset()

    if args.agent_view:
        env = RGBImgPartialObsWrapper(env)
        env = ImgObsWrapper(env)

    window = Window("gym_minigrid")
    window.reg_key_handler(key_handler)

    reset()

    # Blocking event loop
    window.show(block=True)
