import matplotlib.pyplot as plt
from matplotlib import animation
from src.md import MolecularDynamic

def main():
  md = MolecularDynamic()

  fig = plt.figure()
  ax = fig.add_subplot(111)

  l = 20
  ax.set_xlim(-l, l)
  ax.set_ylim(-l, l)

  ax.scatter([], [])

  def update(frame):
    ax.clear()
    ax.set_xlim(-l, l)
    ax.set_ylim(-l, l)
    ax.scatter(md.rx, md.ry)
    md.update()
    return

  anim = animation.FuncAnimation(fig, update, interval=1)

  plt.show()

main()