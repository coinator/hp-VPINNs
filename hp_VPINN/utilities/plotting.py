import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os

def plot(results_dir, x_quad_traing, x_u_train, x_test, u_test, u_pred, grid, error_l2, total_record_l2):
    os.makedirs(results_dir, exist_ok=True)

    x_quad_plot = x_quad_traing
    y_quad_plot = np.empty(len(x_quad_plot))
    y_quad_plot.fill(1)

    x_train_plot = x_u_train
    y_train_plot = np.empty(len(x_train_plot))
    y_train_plot.fill(1)

    font = 24

    fig, ax = plt.subplots()
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize=font)
    plt.ylabel('$error l2\,\, values$', fontsize=font)
    plt.yscale('log')
    plt.grid(True)
    iteration_l2 = [record[0] for record in error_l2]
    loss_his_l2 = [record[1] for record in error_l2]
    plt.plot(iteration_l2, loss_his_l2, 'gray')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig(f'{results_dir}/loss.pdf')

    fig, ax = plt.subplots()
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize=font)
    plt.ylabel('$error inf \,\, values$', fontsize=font)
    plt.yscale('log')
    plt.grid(True)
    iteration_inf = [record[0] for record in total_record_inf]
    loss_his_inf = [record[1] for record in total_record_inf]
    plt.plot(iteration_inf, loss_his_inf, 'gray')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig(f'{results_dir}/loss.pdf')

    fig, ax = plt.subplots()
    plt.tick_params(axis='y', which='both', labelleft='on', labelright='off')
    plt.xlabel('$iteration$', fontsize=font)
    plt.ylabel('$loss l2\,\, values$', fontsize=font)
    plt.yscale('log')
    plt.grid(True)
    iteration_l2 = [record[0] for record in total_record_l2]
    loss_his_l2 = [record[1] for record in total_record_l2]
    plt.plot(iteration_l2, loss_his_l2, 'gray')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig(f'{results_dir}/loss.pdf')


    pnt_skip = 25
    fig, ax = plt.subplots()
    plt.locator_params(axis='x', nbins=6)
    plt.locator_params(axis='y', nbins=8)
    plt.xlabel('$x$', fontsize=font)
    plt.ylabel('$u$', fontsize=font)
    plt.axhline(0, linewidth=0.8, linestyle='-', color='gray')
    for xc in grid:
        plt.axvline(x=xc, linewidth=2, ls='--')

    indexes_1 = np.where(x_test < -0.02)[0][::pnt_skip]
    indexes_2 = np.where(abs(x_test) <= 0.02)[0][::2]
    indexes_3 = np.where(x_test > 0.02)[0][::pnt_skip]
    indexes = np.concatenate((indexes_1, indexes_2, indexes_3), axis=0)

    plt.plot(x_test,
             u_test,
             linewidth=1,
             color='r',
             label=''.join(['$exact$']))
    plt.plot(x_test[indexes], u_pred[indexes], 'k*', label='$VPINN$')
    plt.tick_params(labelsize=20)
    legend = plt.legend(shadow=True, loc='upper left', fontsize=18, ncol=1)
    fig.set_size_inches(w=11, h=5.5)
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
    plt.plot(x_test, abs(u_test - u_pred), 'k')
    plt.tick_params(labelsize=20)
    fig.set_size_inches(w=11, h=5.5)
    plt.savefig(f'{results_dir}/error.pdf')
    plt.show()
