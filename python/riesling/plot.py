import numpy as np
import xarray as xa
import h5py
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import matplotlib.patheffects as effects
from matplotlib.collections import LineCollection
import contextlib
import cmasher
import colorcet
import warnings
from . import io

rc = {'figsize': 4,
      'interpolation': 'none',
      'fontsize':12,
      'effects':([effects.Stroke(linewidth=4, foreground='black'), effects.Normal()])}

def _indexers(D, image, img_zoom, sl, slv, others):
    indexers = {image[0]: img_zoom[0], image[1]: img_zoom[1], sl:slv}
    my_others = others.copy() # Python default argument initialization is DUMB
    my_others |= {k: v for k, v in zip(D.dims, [ii//2 for ii in D.shape]) if k not in (*image, sl, *my_others.keys())}
    indexers |= my_others
    return indexers

def _apply_basis(D, basis_file, basis_tp):
    if basis_file is None:
        return D

    basis = io.read(basis_file)

    if isinstance(basis_tp, int):
        basis = basis[basis_tp:basis_tp+1, :]
    else:
        basis = basis[basis_tp, :]

    data = xa.dot(data, basis, dims=('v', 'v'))
    return data

def planes(fname, pos=0.5, zoom=(slice(None), slice(None)), others={},
           component='mag', clim=None, climp=None, cmap=None, cbar=True,
           rotates=(0, 0, 0), fliplr=False, title=None,
           basis_file=None, basis_tp=0):
    D = _apply_basis(io.read(fname), basis_file, basis_tp)
    posx = int(np.floor(len(D['x'])*pos))
    posy = int(np.floor(len(D['y'])*pos))
    posz = int(np.floor(len(D['z'])*pos))
    data_x = _comp(D.isel(_indexers(D, ('x', 'y'), zoom, 'z', posz, others)), component)
    data_y = _comp(D.isel(_indexers(D, ('x', 'z'), zoom, 'y', posy, others)), component)
    data_z = _comp(D.isel(_indexers(D, ('y', 'z'), zoom, 'x', posx, others)), component)
    clim, cmap = _get_colors(clim, cmap, data_x, component, climp)
    fig, ax = plt.subplots(1, 3, figsize=(rc['figsize']*3, rc['figsize']*1), facecolor='black')

    im_x = _draw(ax[0], _orient(data_x, rotates[0], fliplr), component, clim, cmap)
    im_y = _draw(ax[1], _orient(data_y, rotates[1], fliplr), component, clim, cmap)
    im_z = _draw(ax[2], _orient(data_z, rotates[2], fliplr), component, clim, cmap)
    fig.tight_layout(pad=0)
    _add_colorbar(cbar, component, fig, im_x, clim, title, ax=ax[1])
    plt.close()
    return fig

def slices(fname, image=('x', 'y'), zoom=(slice(None), slice(None)),
           sl='z', n=None, lim=None, others={},
           component='mag', clim=None, climp=None, cmap=None, cbar=True,
           rows=1, rotates=0, fliplr=False, title=None,
           basis_file=None, basis_tp=0):
    D = _apply_basis(io.read(fname), basis_file, basis_tp)
    maxn = len(D[sl])
    if n is None:
        n = maxn
        slv = np.arange(maxn)
    else:
        if lim is None:
            lim = (0, 1)
        slv = np.floor(np.linspace(lim[0]*maxn, lim[1]*maxn, n, endpoint=False)).astype(int)
    data = _comp(D.isel(_indexers(D, image, zoom, sl, slv, others)), component)
    clim, cmap = _get_colors(clim, cmap, data, component, climp)
    cols = int(np.ceil(n / rows))
    fig, all_ax = plt.subplots(rows, cols, figsize=(rc['figsize']*cols, rc['figsize']*rows), facecolor='black')

    for ir in range(rows):
        for ic in range(cols):
            sli = (ir * cols) + ic
            ax = _get_axes(all_ax, ir, ic)
            im = _draw(ax, _orient(data.isel({sl:sli}), rotates, fliplr), component, clim, cmap)
    fig.tight_layout(pad=0)
    _add_colorbar(cbar, component, fig, im, clim, title, ax=all_ax)
    plt.close()
    return fig

def sense(fname, **kwargs):
    return slices(fname, component='x', sl='channel', **kwargs)

def noncart(fname, sample=slice(None), trace=slice(None), **kwargs):
    return slices(fname, component='xlog', sl='channel', image=('sample', 'trace'), zoom=(sample, trace), **kwargs)

def weights(fname, sl_read=slice(None, None, 1), sl_spoke=slice(None, None, 1), log=False, clim=None):
    data = io.read(fname)[sl_spoke, sl_read].T
    if log:
        data = np.log1p(data)
        if clim is None:
            clim = (0, np.max(data))
    elif clim is None:
        clim = np.nanpercentile(np.abs(data), (2, 98))
    ind = np.unravel_index(np.argmax(data, axis=None), data.shape)
    d = data[ind[0], ind[1]]
    fig, ax = plt.subplots(1, 1, figsize=(18, 6), facecolor='w')
    im = ax.imshow(data, interpolation='nearest',
                    cmap='cmr.ember', vmin=clim[0], vmax=clim[1])
    ax.set_xlabel('Spoke')
    ax.set_ylabel('Readout')
    ax.axis('auto')
    fig.colorbar(im, location='right')
    fig.tight_layout()
    plt.close()
    return fig

def diff(fnames, titles=None, image=('x', 'y'), zoom=(slice(None), slice(None)), sl='z', pos=0.5, others={},
         component='mag', clim=None, cmap=None, cbar=False,
         difflim=None, diffmap=None, diffbar=True,
         rotates=0, fliplr=False, title=None,
         basis_files=[None], basis_tps=[0]):

    if len(fnames) < 2:
        raise('Must have more than 1 image to diff')
    if basis_files and len(basis_files) == 1:
        basis_files = basis_files * len(fnames)
    elif not basis_files:
        basis_files = [None,] * len(fnames)
    if basis_tps and len(basis_tps) == 1:
        basis_tps = basis_tps * len(fnames)
    elif not basis_tps:
        basis_tps = [None,] * len(fnames)
    if titles is not None and len(titles) != len(fnames):
        raise('Number of titles and files did not match')

    with contextlib.ExitStack() as stack:
        Ds = [_apply_basis(io.read(fn), f, tp) for fn, f, tp in zip(fnames, basis_files, basis_tps)]
        slv = int(np.floor(len(Ds[0][sl]) * pos))
        data = [_comp(D.isel(_indexers(D, image, zoom, sl, slv, others)), component) for D in Ds]
        data = xa.concat(data, dim='sl')
    ref = abs(data.isel(sl=-1)).max().to_numpy()
    diffs = np.diff(data, n=1, axis=0) * 100 / ref
    n = len(data['sl'])
    clim, cmap = _get_colors(clim, cmap, data, component)
    if component == 'x':
        diff_component = 'x'
    elif component == 'xlog':
        diff_component = 'xlog'
    elif component == 'pha':
        diff_component = 'pha'
    else:
        diff_component = 'real'
    difflim, diffmap = _get_colors(difflim, diffmap, diffs, diff_component)
    fig, ax = plt.subplots(2, n, figsize=(rc['figsize']*n, rc['figsize']*2), facecolor='black')
    for ii in range(n):
        imi = _draw(ax[0, ii], _orient(np.squeeze(data[ii, :, :]), rotates, fliplr), component, clim, cmap)
        if titles is not None:
            ax[0, ii].text(0.1, 0.9, titles[ii], color='white', transform=ax[0, ii].transAxes, ha='left',
                           fontsize=rc['fontsize'], path_effects=rc['effects'])
        if ii > 0:
            imd = _draw(ax[1, ii], _orient(np.squeeze(diffs[ii-1, :, :]), rotates, fliplr), diff_component, difflim, diffmap)
        else:
            ax[1, ii].set_facecolor('black')
            ax[1, ii].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    _add_colorbar(cbar, component, fig, imi, clim, title, ax=ax[0, :])
    _add_colorbar(diffbar, diff_component, fig, imd, difflim, 'Diff (%)', ax=ax[1, :])
    plt.close()
    return fig

def diff_matrix(fnames, titles=None, image=('x', 'y'), zoom=(slice(None), slice(None)), sl='z', pos=0.5, others={},
         component='mag', clim=None, cmap=None, cbar=False,
         difflim=None, diffmap=None, diffbar=True,
         rotates=0, fliplr=False, title=None,
         basis_files=[None], basis_tps=[0]):

    if len(fnames) < 2:
        raise('Must have more than 1 image to diff')
    if basis_files and len(basis_files) == 1:
        basis_files = basis_files * len(fnames)
    elif not basis_files:
        basis_files = [None,] * len(fnames)
    if basis_tps and len(basis_tps) == 1:
        basis_tps = basis_tps * len(fnames)
    elif not basis_tps:
        basis_tps = [None,] * len(fnames)
    if titles is not None and len(titles) != len(fnames):
        raise('Number of titles and files did not match')

    with contextlib.ExitStack() as stack:
        Ds = [_apply_basis(io.read(fn), f, tp) for fn, f, tp in zip(fnames, basis_files, basis_tps)]
        slv = int(np.floor(len(Ds[0][sl]) * pos))
        data = [_comp(D.isel(_indexers(D, image, zoom, sl, slv, others)), component) for D in Ds]
        data = xa.concat(data, dim='sl')
    n = len(data['sl'])
    diffs = []
    ref = abs(data.isel(sl=-1)).max().to_numpy()
    for ii in range(1, n):
        diffs.append([])
        for jj in range(ii):
            diffs[ii - 1].append((data[ii, :, :] - data[jj, :, :]) * 100 / ref)

    clim, cmap = _get_colors(clim, cmap, data, component)
    if component == 'x':
        diff_component = 'x'
    elif component == 'xlog':
        diff_component = 'xlog'
    elif component == 'pha':
        diff_component = 'pha'
    else:
        diff_component = 'real'
    difflim, diffmap = _get_colors(difflim, diffmap, diffs[0][0], diff_component)
    fig, ax = plt.subplots(n, n, figsize=(rc['figsize']*n, rc['figsize']*n), facecolor='black')
    for ii in range(n):
        imi = _draw(ax[ii, ii], _orient(np.squeeze(data[ii, :, :]), rotates, fliplr), component, clim, cmap)
        if titles is not None:
            ax[ii, ii].text(0.5, 0.05, titles[ii], color='white', transform=ax[ii, ii].transAxes, ha='center',
                            fontsize=rc['fontsize'], path_effects=rc['effects'])
        for jj in range(ii):
            imd = _draw(ax[jj, ii], _orient(np.squeeze(diffs[ii - 1][jj]), rotates, fliplr), diff_component, difflim, diffmap)
        for jj in range(ii, n):
            ax[jj, ii].set_facecolor('black')
            ax[jj, ii].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    _add_colorbar(cbar, component, fig, imi, clim, title, ax=ax[0, 0])
    _add_colorbar(diffbar, diff_component, fig, imd, difflim, 'Diff (%)', ax=ax[0, -1])
    plt.close()
    return fig

def _comp(data, component):
    if component == 'x':
        pass
    elif component == 'xlog':
        pass
    elif component == 'mag':
        data = np.abs(data)
    elif component == 'log':
        data = np.log1p(np.abs(data))
    elif component == 'pha':
        data = np.angle(data)
    elif component == 'real':
        data = np.real(data)
    elif component == 'imag':
        data = np.imag(data)
    else:
        data = np.real(data)
        warnings.warn('Unknown component, taking real')
    return data

def _orient(img, rotates, fliplr=False):
    if rotates > 0:
        img = np.rot90(img, rotates)
    if fliplr:
        img = np.fliplr(img)
    return img

def _symmetrize_real(x):
    x[1] = np.amax([np.abs(x[0]), np.abs(x[1])])
    x[0] = -x[1]
    return x

def _get_colors(clim, cmap, img, component, climp=None):
    if not clim:
        if climp is None:
            climp= (2, 99)
        if component == 'mag':
            clim = np.nanpercentile(img, climp)
        elif component == 'log':
            clim = np.nanpercentile(img, climp)
        elif component == 'pha':
            clim = (-np.pi, np.pi)
        elif component == 'real':
            clim = _symmetrize_real(np.nanpercentile(img, climp))
        elif component == 'imag':
            clim = _symmetrize_real(np.nanpercentile(img, climp))
        elif component == 'x':
            clim = np.nanpercentile(np.abs(img), climp)
        elif component == 'xlog':
            clim = np.nanpercentile(np.log1p(np.abs(img)), climp)
            if not clim.any():
                clim = np.nanpercentile(np.log1p(np.abs(img)), (0, 100))
        else:
            raise(f'Unknown component {component}')
    if clim[0] == clim[1]:
        print(f'Color limits were {clim}, expanding')
        clim[1] = clim[0] + 1
    if not cmap:
        if component == 'mag':
            cmap = 'gray'
        elif component == 'log':
            cmap = 'gray'
        elif component == 'real':
            cmap = 'cmr.iceburn'
        elif component == 'imag':
            cmap = 'cmr.iceburn'
        elif component == 'pha':
            cmap = 'cet_colorwheel'
        elif component == 'x':
            cmap = 'cet_colorwheel'
        elif component == 'xlog':
            cmap = 'cet_colorwheel'
        else:
            raise(f'Unknown component {component}')
    return (clim, cmap)

def _add_colorbar(cbar, component, fig, im, clim, title, ax=None, cax=None, vpos='bottom'):
    if not cbar:
        if title is not None:
            fig.text(0.5, 0.95, title, color='white', ha='center', fontsize=rc['fontsize'], path_effects=rc['effects'])
    elif component == 'x' or component == 'xlog':
        _add_colorball(clim, ax=ax, cax=cax)
        if title is not None:
            fig.text(0.5, 0.95, title, color='white', ha='center', fontsize=rc['fontsize'], path_effects=rc['effects'])
    else:
        if cax is None:
            ax = _first(ax)
            sz = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).size
            height = 1.4 * rc['fontsize'] / (sz[1] * 72) # 72 points per inch
            if vpos == 'bottom':
                vpos = 0.05
            elif vpos == 'top':
                vpos = 0.095 - height
            cax = _first(ax).inset_axes(bounds=(0.1, vpos, 0.8, height), facecolor='black')
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        axes = cb.ax
        ticks = (clim[0], np.sum(clim)/2, clim[1])
        labels = (f' {clim[0]:2.2f}', title, f'{clim[1]:2.2f} ')
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels, fontsize=rc['fontsize'], path_effects=rc['effects'])
        cb.ax.tick_params(axis='x', bottom=False, top=False)
        cb.ax.get_xticklabels()[0].set_ha('left')
        cb.ax.get_xticklabels()[1].set_ha('center')
        cb.ax.get_xticklabels()[2].set_ha('right')
        cb.ax.tick_params(color='w', labelcolor='w', pad=-1.3*rc['fontsize'])

def _add_colorball(clim, ax=None, cax=None, cmap='cet_colorwheel'):
    if cax is None:
        ax = _first(ax)
        sz = ax.get_window_extent().transformed(ax.figure.dpi_scale_trans.inverted()).size
        w = 3 * 1.4 * rc['fontsize'] / (sz[0] * 72) # 72 points per inch
        h = 3 * 1.4 * rc['fontsize'] / (sz[1] * 72) # 72 points per inch
        cax = ax.inset_axes(bounds=(0.1*w, 0.1*h, h, h), projection='polar', facecolor='black')
    theta, rad = np.meshgrid(np.linspace(-np.pi, np.pi, 64), np.linspace(0, 1, 64))
    ones = np.ones_like(theta)
    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    colorized = smap.to_rgba(list(theta.ravel()), alpha=1., bytes=False)
    for ii in range(len(colorized)):
        for ij in range(3):
            colorized[ii][ij] = colorized[ii][ij] * rad.ravel()[ii]
    quads = cax.pcolormesh(theta, rad, ones, shading='nearest', color=colorized)
    cax.grid(visible=True, linewidth=2, color='white')
    cax.tick_params(axis='x', colors='white')
    cax.tick_params(axis='y', colors='white')
    cax.spines[:].set_color('white')
    cax.spines[:].set_linewidth(2)
    cax.spines[:].set_visible(True)
    cax.set_xticks([0, np.pi/2])
    cax.set_xticklabels([f'{clim[1]:.0e}', f'{clim[1]:.0e}' + 'i'],
                        fontsize=rc['fontsize'], path_effects=rc['effects'])
    cax.get_xticklabels()[0].set_ha('left')
    cax.get_xticklabels()[1].set_va('bottom')
    cax.xaxis.set_tick_params(pad=-0.5*rc['fontsize'])
    cax.set_yticks([0, 1])
    cax.set_yticklabels([])

def _first(maybe_iterable):
    """ Terrible hack to support the fact subplots can return an axis, an 
        array of axes, or an array of an array of axes """
    if hasattr(maybe_iterable, "__len__"):
        while hasattr(maybe_iterable, "__len__"):
            maybe_iterable = maybe_iterable[0]
        return maybe_iterable
    else:
        return maybe_iterable

def _get_axes(ax, ir, ic):
    if hasattr(ax, "shape"):
        if len(ax.shape) == 2:
            return ax[ir, ic]
        else:
            return ax[ic]
    else:
        return ax

def _draw(ax, img, component, clim, cmap):
    if component == 'x':
        _draw_x(ax, img, clim, cmap)
        return None
    elif component == 'xlog':
        _draw_x(ax, img, clim, cmap, True)
        return None
    else:
        im = ax.imshow(img, cmap=cmap, interpolation=rc['interpolation'], vmin=clim[0], vmax=clim[1])
        ax.axis('off')
        return im

def _draw_x(ax, img, clim, cmap='cet_colorwheel', log=False):
    mag = np.real(np.abs(img))
    if log:
        mag = np.log1p(mag);
    mag = np.clip((mag - clim[0]) / (clim[1] - clim[0]), 0, 1)

    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    pha = np.angle(img)
    colorized = smap.to_rgba(pha, alpha=1., bytes=False)[:, :, 0:3] * mag.to_numpy()[..., np.newaxis]
    ax.imshow(colorized, interpolation=rc['interpolation'])
    ax.axis('off')

def basis(fname, sl_spoke=slice(None), b=slice(None)):
    basis = io.read(fname)[sl_spoke,b]
    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    ax[0].plot(np.real(basis))
    ax[1].plot(np.imag(basis))
    leg = [str(x) for x in range(basis.shape[1])]
    ax[0].legend(leg)
    ax[0].grid(True)
    # ax[0].autoscale(enable=True, tight=True)
    ax[1].grid(True)
    # ax[1].autoscale(enable=True, tight=True)
    plt.close()
    return fig

def dynamics(filename, sl=slice(None), vlines=None):
    with h5py.File(filename) as f:
        dyn = f['dynamics'][sl,:]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(dyn.T)
        ax.grid('on')
        if vlines:
            [ax.axvline(x) for x in vlines]
        plt.close()
        return fig

def dictionary(filename):
    with h5py.File(filename) as f:
        d = f['dictionary']
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')
        ax.scatter(xs = d[:, 0], ys = d[:, 1], zs = d[:, 2])
        plt.close()
        return fig

def traj2d(filename, read_slice=slice(None), spoke_slice=slice(None), color='read', sps=None, zoom=False):
    with h5py.File(filename) as f:
        traj = np.array(f['trajectory'])
        if color == 'read':
            c = np.tile(np.arange(len(traj[0, read_slice, 0])), len(traj[spoke_slice, 0, 0]))
        elif color == 'seg':
            c = np.tile(np.repeat(np.arange(sps),
                                  len(traj[0, read_slice, 0])),
                        int(len(traj[spoke_slice, 0, 0])/sps))
        else:
            c = np.tile(np.arange(len(traj[spoke_slice, 0, 0])), (len(traj[0, read_slice, 0]), 1)).ravel(order='F')
        nd = traj.shape[-1]
        fig, ax = plt.subplots(1, nd, figsize=(12, 4), facecolor='w')
        for ii in range(nd):
            ax[ii].grid()
            ax[ii].scatter(traj[spoke_slice, read_slice, ii % nd],
                       traj[spoke_slice, read_slice, (ii + 1) % nd],
                       c=c, cmap='cmr.ember', s=0.5)
            ax[ii].set_aspect('equal')
            if not zoom:
                ax[ii].set_xlim((-0.5,0.5))
                ax[ii].set_ylim((-0.5,0.5))
        fig.tight_layout()
        plt.close()
    return fig

def traj3d(filename, read_slice=slice(None), spoke_slice=slice(None), color='read', sps=None,
           angles=[30,-60,0], zoom=False, draw_axes=False):
    with h5py.File(filename) as ff:
        traj = ff['trajectory'][spoke_slice, read_slice, :]
        if color == 'read':
            c = np.tile(np.arange(traj.shape[1]), (traj.shape[0]))
        elif color == 'seg':
            c = np.tile(np.repeat(np.arange(sps), traj.shape[1]), int(traj.shape[0]/sps))
        else:
            c = np.tile(np.arange(traj.shape[0]), (traj.shape[1], 1))
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')

        if draw_axes:
            x, y, z = np.array([[-0.5,0,0],[0,-0.5,0],[0,0,-0.5]])
            u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
            ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1)
        ax.scatter(traj[:, :, 0], traj[:, :, 1], traj[:, :, 2],
                c=c, s=3, cmap='cmr.ember')
        if not zoom:
            ax.set_xlim((-0.5,0.5))
            ax.set_ylim((-0.5,0.5))
            ax.set_zlim((-0.5,0.5))
        ax.view_init(elev=angles[0], azim=angles[1], vertical_axis='z')
        fig.tight_layout()
        plt.close()
    return fig
