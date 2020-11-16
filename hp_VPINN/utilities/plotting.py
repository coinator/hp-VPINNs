from hp_VPINN.utilities import np

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot(x_quadrature, x_boundary, x_prediction, u_prediction, u_correct,
         total_record, grid, results_dir=None):
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
    if results_dir:
        plt.savefig(f'{results_dir}/Train-Quad-pnts.pdf')

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
    if results_dir:
        plt.savefig(f'{results_dir}/loss.pdf')

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
    if results_dir:
        plt.savefig(f'{results_dir}/error.pdf')

    pnt_skip = 25
    fig, ax = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize=font)
    plt.ylabel('$u$', fontsize=font)
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls='--')

    indexes_1 = np.where(x_prediction < -0.02)[0][::pnt_skip]
    indexes_2 = np.where(abs(x_prediction) <= 0.02)[0][::2]
    indexes_3 = np.where(x_prediction > 0.02)[0][::pnt_skip]
    indexes = np.concatenate((indexes_1, indexes_2, indexes_3), axis=0)

    plt.plot(x_prediction[indexes],
             u_correct[indexes],
             linewidth=1,
             color='r',
             label=''.join(['$exact$']))
    plt.plot(x_prediction[indexes],
             u_prediction[indexes],
             'k*',
             label='$VPINN$')
    plt.tick_params(labelsize=20)
    legend = plt.legend(shadow=True, loc='upper left', fontsize=18, ncol=1)
    fig.set_size_inches(w=11, h=5.5)
    if results_dir:
        plt.savefig(f'{results_dir}/prediction.pdf')

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
    if results_dir:
        plt.savefig(f'{results_dir}/point_wiseerror.pdf')

    plt.show()
