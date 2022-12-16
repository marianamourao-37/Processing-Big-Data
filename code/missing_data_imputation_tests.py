import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random as r


## Just some tests for the imputing missing data


def pn():
    return 1 if r.random() < 0.5 else -1


def create_skel(xi, yi, xvar, yvar, random):
    '''Creates a simple skel'''
    skel = [[xi + 0 * xvar + random * r.random() * pn(), yi + 0 * yvar + random * r.random() * pn()],
            [xi + 4 * xvar + random * r.random() * pn(), yi + 0 * yvar + random * r.random() * pn()],
            [xi + 1 * xvar + random * r.random() * pn(), yi + 1 * yvar + random * r.random() * pn()],
            [xi + 3 * xvar + random * r.random() * pn(), yi + 1 * yvar + random * r.random() * pn()],
            [xi + 2 * xvar + random * r.random() * pn(), yi + 2 * yvar + random * r.random() * pn()],
            [xi + 2 * xvar + random * r.random() * pn(), yi + 5 * yvar + random * r.random() * pn()],
            [xi + 0 * xvar + random * r.random() * pn(), yi + 3 * yvar + random * r.random() * pn()],
            [xi + 1 * xvar + random * r.random() * pn(), yi + 4 * yvar + random * r.random() * pn()],
            [xi + 3 * xvar + random * r.random() * pn(), yi + 4 * yvar + random * r.random() * pn()],
            [xi + 4 * xvar + random * r.random() * pn(), yi + 5 * yvar + random * r.random() * pn()],
            [xi + 2 * xvar + random * r.random() * pn(), yi + 6 * yvar + random * r.random() * pn()],
            [xi + 1.5 * xvar + random * r.random() * pn(), yi + 7 * yvar + random * r.random() * pn()],
            [xi + 2.5 * xvar + random * r.random() * pn(), yi + 7 * yvar + random * r.random() * pn()]]
    return np.array(skel)


def aglom_skels(n):
    '''Randomizes and creates n skels'''
    skels = []
    plt.xlim([-20, 20 + 4 * 0.6])
    plt.ylim([-20, 20 + 2.5 * 7])
    for _ in range(n):
        xi = r.random() * 20 * pn()
        yi = r.random() * 20 * pn()
        xvar = 0.5 + 0.1 * r.random()
        yvar = 1.75 + 0.25 * r.random()
        random = 0.1 * r.random()
        skels += [create_skel(xi, yi, xvar, yvar, random)]
    return np.array(skels)

def imput_missing(skels, perc):
    '''Impute missing values in complete data'''
    for skel in skels:
        for joint in skel:
            if r.random() < perc:
                pass
                # TO DO


def plot_skels(skels):
    '''Draws skels in a scatter plot with all different colours'''
    colours = cm.rainbow(np.linspace(0, 1, len(skels)))
    for i, skel in enumerate(skels):
        for joint in skel:
            plt.scatter(joint[0], joint[1], color=colours[i], marker='.')
    plt.show()


plot_skels(aglom_skels(9))


