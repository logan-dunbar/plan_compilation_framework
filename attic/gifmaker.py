import glob
import bezier
import numpy as np

from generativepy.movie import MovieBuilder
from generativepy.drawing import setup, make_image_frames
from generativepy.color import Color
from generativepy.geometry import Circle, Image
from generativepy.gif import save_animated_gif
from PIL import Image as PImage

def make_gif():
  # folder = 'images/q_optimistic'
  # folder = 'images/q_pessimistic'
  # folder = 'images/q_planner'
  # folder = 'images/q_planner_pessimistic'
  folder = 'images/pc_learner'
  # folder = 'images/pc_explorer'
  # folder = 'images/temp'

  # agent = 'q_opt'
  # agent = 'q_pess'
  # agent = 'q_plan_opt'
  # agent = 'q_plan_pess'
  agent = 'pc'

  prefix = 'e'
  # prefix = 'value'

  im_paths = list(sorted(glob.glob(f'{folder}/{prefix}_*.png')))

  fps = 30
  n_frames = min(len(im_paths), 1000)

  if prefix == 'e':
    width = 330
    height = 330
    scale = 1
  else:
    im1 = PImage.open(im_paths[0])
    width = min(im1.width, 500)
    height = min(im1.height, 500)
    scale = 0.5

  # frames = []
  # for im_path in glob.glob("images/planner_*.png"):
  #   im_frame = Image.open(im_path)
  #   frames.append(np.array(im_frame.getdata()))

  def draw(ctx, width, height, frame_no, frame_count):
    setup(ctx, width, height, background=Color(0.8))
    Image(ctx).of_file_position(im_paths[frame_no], (0, 0)).scale(scale).paint()

  frames = make_image_frames(draw, width, height, n_frames)
  save_animated_gif(f'{agent}_{prefix}.gif', frames, 1. / fps, loop=0)

  # movie_builder = MovieBuilder(fps)
  # movie_builder.add_scene((frames, t))
  # movie_builder.make_movie(f'{name}.mp4')

  debug = 0


if __name__ == '__main__':
  make_gif()

  # im_paths = list(sorted(glob.glob(f'images/q_optimistic/*.jpg')))
  #
  # for im_path in im_paths:
  #   im1 = PImage.open(im_path)
  #   im1.save(im_path.replace('.jpg', '.png'))
