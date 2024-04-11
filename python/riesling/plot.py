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

def _symmetrize(x):
    m = np.max(x)
    return [-m, m]

def _cmap(component):
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
    return cmap

def _clim(component, climp, data):
    if climp is None:
        climp = (2, 99)
    if component == 'mag':
        clim = np.nanpercentile(data, climp)
    elif component == 'log':
        clim = np.nanpercentile(data, climp)
    elif component == 'pha':
        clim = (-np.pi, np.pi)
    elif component == 'real':
        clim = _symmetrize(np.nanpercentile(np.abs(data), climp))
    elif component == 'imag':
        clim = _symmetrize(np.nanpercentile(np.abs(data), climp))
    elif component == 'x':
        clim = np.nanpercentile(np.abs(data), climp)
    elif component == 'xlog':
        clim = np.nanpercentile(np.log1p(np.abs(data)), climp)
        if not clim.any():
            clim = np.nanpercentile(np.log1p(np.abs(data)), (0, 100))
    else:
        raise(f'Unknown component {component}')
    return clim

def _comp(data, component, cmap, clim, climp):
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

    if cmap is None:
        cmap = _cmap(component)

    if clim is None:
        clim = _clim(component, climp, data)
    if clim[0] == clim[1]:
        print(f'Color limits were {clim}, expanding')
        clim[1] = clim[0] + 1

    return data, cmap, clim

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

def _orient(img, rotates, fliplr=False):
    img = img.to_numpy()
    if rotates != 0:
        img = np.rot90(img, rotates)
    if fliplr:
        img = np.fliplr(img)
    return img

def planes(fname, pos=0.5, zoom=(slice(None), slice(None)), others={},
           component='mag', clim=None, climp=None, cmap=None, cbar=True,
           rotates=(0, 0, 0), fliplr=False, title=None,
           basis_file=None, basis_tp=0):
    D = _apply_basis(io.read_data(fname), basis_file, basis_tp)
    posx = int(np.floor(len(D['x'])*pos))
    posy = int(np.floor(len(D['y'])*pos))
    posz = int(np.floor(len(D['z'])*pos))
    data_x, cmap, clim = _comp(D.isel(_indexers(D, ('x', 'y'), zoom, 'z', posz, others)), component, cmap, clim, climp)
    data_y, _, _ = _comp(D.isel(_indexers(D, ('x', 'z'), zoom, 'y', posy, others)), component, cmap, clim, climp)
    data_z, _, _ = _comp(D.isel(_indexers(D, ('y', 'z'), zoom, 'x', posx, others)), component, cmap, clim, climp)
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
    D = _apply_basis(io.read_data(fname), basis_file, basis_tp)
    maxn = len(D[sl])
    if n is None:
        n = maxn
        slis = np.arange(maxn)
    else:
        if lim is None:
            lim = (0, 1)
        slis = np.floor(np.linspace(lim[0]*maxn, lim[1]*maxn, n, endpoint=False)).astype(int)
    data, cmap, clim = _comp(D.isel(_indexers(D, image, zoom, sl, slis, others)), component, cmap, clim, climp)
    cols = int(np.ceil(n / rows))
    fig, all_ax = plt.subplots(rows, cols, figsize=(rc['figsize']*cols, rc['figsize']*rows), facecolor='black')

    for ir in range(rows):
        for ic in range(cols):
            ii = (ir * cols) + ic
            ax = _get_axes(all_ax, ir, ic)
            if ii < len(slis):
                im = _draw(ax, _orient(data.isel({sl:ii}), rotates, fliplr), component, clim, cmap)
            else:
                ax.set_facecolor('k')
    fig.tight_layout(pad=0)
    _add_colorbar(cbar, component, fig, im, clim, title, ax=all_ax)
    plt.close()
    return fig

def sense(fname, **kwargs):
    return slices(fname, component='x', sl='channel', **kwargs)

def noncart(fname, sample=slice(None), trace=slice(None), **kwargs):
    return slices(fname, component='xlog', sl='channel', image=('sample', 'trace'), zoom=(sample, trace), **kwargs)

def weights(fname, sl_read=slice(None, None, 1), sl_spoke=slice(None, None, 1), log=False, clim=None):
    data = io.read(fname)[sl_spoke, sl_read].values.T
    if log:
        data = np.log1p(data)
        if clim is None:
            clim = (0, np.max(data))
    elif clim is None:
        clim = np.nanpercentile(np.abs(data), (2, 98))
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

def _diff(fnames, titles=None, image=('x', 'y'), zoom=(slice(None), slice(None)), sl='z', pos=0.5, others={},
         component='mag', clim=None, climp=None, cmap=None, cbar=False,
         difflim=None, diffmap=None, diffbar=True, alldiffs=True, percentdiffs=True,
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
        Ds = [_apply_basis(io.read_data(fn), f, tp) for fn, f, tp in zip(fnames, basis_files, basis_tps)]
        slv = int(np.floor(len(Ds[0][sl]) * pos))
        all = [_comp(D.isel(_indexers(D, image, zoom, sl, slv, others)), component, cmap, clim, climp) for D in Ds]
        data = [a[0] for a in all]
        cmaps = [a[1] for a in all]
        clims = [a[2] for a in all]
        data = xa.concat(data, dim='sl')
    n = len(data['sl'])
    diffs = []
    ref = abs(data.isel(sl=-1)).max()

    if component == 'x':
        diffcomp = 'x'
    elif component == 'xlog':
        diffcomp = 'xlog'
    elif component == 'pha':
        diffcomp = 'pha'
    else:
        diffcomp = 'real'

    for ii in range(1, n):
        if alldiffs:
            diffs.append([])
            for jj in range(ii):
                d = (data[ii, :, :] - data[jj, :, :])
                if percentdiffs:
                    d = d * 100 / ref
                diffs[ii - 1].append(d)
        else:
            d = (data[ii, :, :] - data[ii - 1, :, :])
            if percentdiffs:
                d = d * 100 / ref
            diffs.append(d)

    if difflim is None:
        if alldiffs:
            difflim = _clim(diffcomp, None, diffs[0][0])
        else:
            difflim = _clim(diffcomp, None, diffs[0])
    diffmap = _cmap(diffcomp)
    return n, data, diffs, cmaps[0], clims[0], diffcomp, diffmap, difflim


def diff(fnames, titles=None, image=('x', 'y'), zoom=(slice(None), slice(None)), sl='z', pos=0.5, others={},
         component='mag', clim=None, climp=None, cmap=None, cbar=False,
         difflim=None, diffmap=None, diffbar=True, percentdiffs=True,
         rotates=0, fliplr=False, title=None,
         basis_files=[None], basis_tps=[0]):
    n, data, diffs, cmap, clim, diffcomp, diffmap, difflim = _diff(fnames, titles, image, zoom, sl, pos, others,
                                                                   component, clim, climp, cmap, cbar,
                                                                   difflim, diffmap, diffbar, False, percentdiffs,
                                                                   rotates, fliplr, title, basis_files, basis_tps)
    fig, ax = plt.subplots(2, n, figsize=(rc['figsize']*n, rc['figsize']*2), facecolor='black')
    for ii in range(n):
        imi = _draw(ax[0, ii], _orient(np.squeeze(data[ii, :, :]), rotates, fliplr), component, clim, cmap)
        if titles is not None:
            ax[0, ii].text(0.1, 0.9, titles[ii], color='white', transform=ax[0, ii].transAxes, ha='left',
                           fontsize=rc['fontsize'], path_effects=rc['effects'])
        if ii > 0:
            imd = _draw(ax[1, ii], _orient(np.squeeze(diffs[ii-1]), rotates, fliplr), diffcomp, difflim, diffmap)
        else:
            ax[1, ii].set_facecolor('black')
            ax[1, ii].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    _add_colorbar(cbar, component, fig, imi, clim, title, ax=ax[0, :])
    if percentdiffs:
        _add_colorbar(diffbar, diffcomp, fig, imd, difflim, 'Diff (%)', ax=ax[0, -1])
    else:
        _add_colorbar(diffbar, diffcomp, fig, imd, difflim, 'Difference', ax=ax[0, -1])
    plt.close()
    return fig

def diff_matrix(fnames, titles=None, image=('x', 'y'), zoom=(slice(None), slice(None)), sl='z', pos=0.5, others={},
         component='mag', clim=None, climp=None, cmap=None, cbar=False,
         difflim=None, diffmap=None, diffbar=True, percentdiffs=True,
         rotates=0, fliplr=False, title=None,
         basis_files=[None], basis_tps=[0]):
    n, data, diffs, cmap, clim, diffcomp, diffmap, difflim = _diff(fnames, titles, image, zoom, sl, pos, others,
                                                                   component, clim, climp, cmap, cbar,
                                                                   difflim, diffmap, diffbar, True, percentdiffs,
                                                                   rotates, fliplr, title, basis_files, basis_tps)

    fig, ax = plt.subplots(n, n, figsize=(rc['figsize']*n, rc['figsize']*n), facecolor='black')
    for ii in range(n):
        imi = _draw(ax[ii, ii], _orient(np.squeeze(data[ii, :, :]), rotates, fliplr), component, clim, cmap)
        if titles is not None:
            ax[ii, ii].text(0.5, 0.05, titles[ii], color='white', transform=ax[ii, ii].transAxes, ha='center',
                            fontsize=rc['fontsize'], path_effects=rc['effects'])
        for jj in range(ii):
            imd = _draw(ax[jj, ii], _orient(np.squeeze(diffs[ii - 1][jj]), rotates, fliplr), diffcomp, difflim, diffmap)
        for jj in range(ii, n):
            ax[jj, ii].set_facecolor('black')
            ax[jj, ii].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)
    _add_colorbar(cbar, component, fig, imi, clim, title, ax=ax[0, 0])
    if percentdiffs:
        _add_colorbar(diffbar, diffcomp, fig, imd, difflim, 'Diff (%)', ax=ax[0, -1])
    else:
        _add_colorbar(diffbar, diffcomp, fig, imd, difflim, 'Difference', ax=ax[0, -1])
    plt.close()
    return fig

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
    colorized = smap.to_rgba(pha, alpha=1., bytes=False)[:, :, 0:3] * mag[..., np.newaxis]
    ax.imshow(colorized, interpolation=rc['interpolation'])
    ax.axis('off')

def _traj_color(traj, color, seg_length):
    if color == 'sample':
        return np.tile(np.arange(traj.shape[1]), (traj.shape[0]))
    elif seg_length is not None:
        return np.tile(np.repeat(np.arange(seg_length), traj.shape[1]), traj.shape[0]//seg_length)
    else:
        return np.repeat(np.arange(traj.shape[0]), traj.shape[1])

def traj2d(filename, sample=slice(None), trace=slice(None), color='trace', seg_length=None):
    with h5py.File(filename) as f:
        traj = np.array(f['trajectory'][trace, sample, :])
        c = _traj_color(traj, color, seg_length)
        nd = traj.shape[-1]
        fig, ax = plt.subplots(1, nd, figsize=(12, 4), facecolor='w')
        for ii in range(nd):
            ax[ii].grid()
            ax[ii].scatter(traj[:, :, ii % nd],
                       traj[:, :, (ii + 1) % nd],
                       c=c, cmap='cmr.ember', s=0.5)
            ax[ii].set_aspect('equal')
        fig.tight_layout()
        plt.close()
    return fig

def traj3d(filename, sample=slice(None), trace=slice(None), color='trace', seg_length=None,
           angles=[30,-60,0], draw_axes=False):
    with h5py.File(filename) as ff:
        traj = ff['trajectory'][trace, sample, :]
        c = _traj_color(traj, color, seg_length)
        fig = plt.figure(figsize=(6,6))
        ax = fig.add_subplot(projection='3d')

        if draw_axes:
            x, y, z = np.array([[-0.5,0,0],[0,-0.5,0],[0,0,-0.5]])
            u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
            ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1)
        ax.scatter(traj[:, :, 0], traj[:, :, 1], traj[:, :, 2],
                c=c, s=3, cmap='cmr.ember')
        ax.view_init(elev=angles[0], azim=angles[1], vertical_axis='z')
        fig.tight_layout()
        plt.close()
    return fig
