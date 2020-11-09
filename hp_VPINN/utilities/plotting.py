from hp_VPINN.utilities import np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(x_quadrature, x_boundary, x_prediction, u_prediction, u_correct,
         total_record, grid):
    y_quad_plot = np.ones(len(x_quadrature))
    y_train_plot = np.ones(len(x_boundary))

    fig = plt.figure(0)
    gridspec.GridSpec(3, 1)

    plt.subplot2grid((3, 1), (0, 0))
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.yticks([])
    plt.title('$Quadrature \,\, Points$')
    plt.xlabel('$x$')
    plt.axhline(1, linewidth=1, linestyle='-', color='red')
    plt.axvline(-1, linewidth=1, linestyle='--', color='red')
    plt.axvline(1, linewidth=1, linestyle='--', color='red')
    plt.scatter(x_quadrature, y_quad_plot, color='green')

    plt.subplot2grid((3, 1), (1, 0))
    plt.tight_layout()
    plt.locator_params(axis='x', nbins=6)
    plt.yticks([])
    plt.title('$Training \,\, Points$')
    plt.xlabel('$x$')
    plt.axhline(1, linewidth=1, linestyle='-', color='red')
    plt.axvline(-1, linewidth=1, linestyle='--', color='red')
    plt.axvline(1, linewidth=1, linestyle='--', color='red')
    plt.scatter(x_boundary, y_train_plot, color='blue')

    fig.tight_layout()
    fig.set_size_inches(w=10, h=7)
    plt.savefig('Results/Train-Quad-pnts.pdf')

    font = 24

    fig, ax = plt.subplots()
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize=font)
    plt.ylabel('$loss \,\, values$', fontsize=font)
    plt.yscale('log')
    plt.grid(True)
    iteration = [total_record[i][0] for i in range(len(total_record))]
    loss_his = [total_record[i][1] for i in range(len(total_record))]
    plt.plot(iteration, loss_his, 'gray')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig('Results/loss.pdf')

    fig, ax = plt.subplots()
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize=font)
    plt.ylabel('$error \,\, values$', fontsize=font)
    plt.yscale('log')
    plt.grid(True)
    iteration = [total_record[i][0] for i in range(len(total_record))]
    error_his = [total_record[i][2] for i in range(len(total_record))]
    plt.plot(iteration, error_his, 'gray')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig('Results/error.pdf')

    pnt_skip = 25
    fig, ax = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize=font)
    plt.ylabel('$u$', fontsize=font)
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls='--')
    plt.plot(x_prediction,
             u_correct,
             linewidth=1,
             color='r',
             label=''.join(['$exact$']))
    plt.plot(x_prediction[0::pnt_skip],
             u_prediction[0::pnt_skip],
             'k*',
             label='$VPINN$')
    plt.tick_params(labelsize=20)
    legend = plt.legend(shadow=True, loc='upper left', fontsize=18, ncol=1)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig('Results/prediction.pdf')

    fig, ax = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize=font)
    plt.ylabel('point-wise error', fontsize=font)
    plt.yscale('log')
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls='--')
    plt.plot(x_prediction, abs(u_correct - u_prediction), 'k')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig('Results/error.pdf')

    plt.show()
