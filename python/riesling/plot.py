import numpy as np
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

rc = {'figsize': 4,
      'interpolation': 'none',
      'fontsize':12,
      'effects':([effects.Stroke(linewidth=2, foreground='black'), effects.Normal()])}

def slices(fname, dset='image', n=4, axis='z', start=0.25, stop=0.75,
           other_dims=[], other_indices=[], img_offset=-1, img_slices=(slice(None), slice(None)),
           component='mag', clim=None, cmap=None, cbar=True,
           rows=1, rotates=0, fliplr=False, title=None):

    slice_dim, img_dims = _get_dims(axis, img_offset)
    with h5py.File(fname, 'r') as f:
        D = f[dset]
        maxn = D.shape[slice_dim]
        n = np.amin([n, maxn])
        slices = np.floor(np.linspace(start*maxn, stop*maxn, n, endpoint=True)).astype(int)
        data = _get_slices(D, slice_dim, slices, img_dims, img_slices, other_dims, other_indices)

    clim, cmap = _get_colors(clim, cmap, data, component)
    cols = int(np.ceil(n / rows))
    fig, all_ax = plt.subplots(rows, cols, figsize=(rc['figsize']*cols, rc['figsize']*rows), facecolor='black')

    for ir in range(rows):
        for ic in range(cols):
            sl = (ir * cols) + ic
            ax = _get_axes(all_ax, ir, ic)
            im = _draw(ax, _orient(np.squeeze(data[sl, :, :]), rotates, fliplr), component, clim, cmap)
    fig.tight_layout(pad=0)
    _add_colorbar(cbar, component, fig, all_ax, im, clim, title)
    plt.close()
    return fig

def series(fname, dset='image', axis='z', slice_pos=0.5, series_dim=-1, series_slice=slice(None),
           other_dims=[], other_indices=[], img_offset=-1, img_slices=(slice(None), slice(None)),
           component='mag', clim=None, cmap=None, cbar=True, rows=1, rotates=0, fliplr=False, title=None):

    slice_dim, img_dims = _get_dims(axis, img_offset)
    with h5py.File(fname, 'r') as f:
        D = f[dset]
        slice_index = int(np.floor(D.shape[slice_dim] * slice_pos))
        data = _get_slices(D, series_dim, series_slice, img_dims, img_slices, [slice_dim, *other_dims], [slice_index, *other_indices])

    n = data.shape[-3]
    clim, cmap = _get_colors(clim, cmap, data, component)
    cols = int(np.ceil(n / rows))
    fig, all_ax = plt.subplots(rows, cols, figsize=(rc['figsize']*cols, rc['figsize']*rows), facecolor='black')
    for ir in range(rows):
        for ic in range(cols):
            sl = (ir * cols) + ic
            ax = _get_axes(all_ax, ir, ic)
            im = _draw(ax, _orient(np.squeeze(data[sl, :, :]), rotates, fliplr), component, clim, cmap)
    fig.tight_layout(pad=0)
    _add_colorbar(cbar, component, fig, im, clim, title, ax=all_ax)
    plt.close()
    return fig

def sense(fname, **kwargs):
    return series(fname, dset='sense', component='x', **kwargs)

def diff(fnames, dsets=['image'], titles=None, axis='z', slice_pos=0.5,
         other_dims=[], other_indices=[], img_offset=-1, img_slices=(slice(None), slice(None)),
         component='mag', clim=None, cmap=None, cbar=True,
         diff_component='real', difflim=None, diffmap=None,
         rotates=0, fliplr=False, title=None):

    if len(dsets) == 1:
        dsets = dsets * len(fnames)
    if titles is not None and len(titles) != len(fnames):
        raise('Number of titles and files did not match')

    slice_dim, img_dims = _get_dims(axis, img_offset)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fn, 'r')) for fn in fnames]
        dsets = [f[dset] for f, dset in zip(files, dsets)]
        slice_index = int(np.floor(dsets[0].shape[slice_dim] * slice_pos))
        data = [_get_slices(D, slice_dim, slice(slice_index, slice_index+1), img_dims, img_slices, other_dims, other_indices) for D in dsets]
        data = np.concatenate(data)
    diffs = np.diff(data, n=1, axis=0) / data[:-1, :, :]
    n = data.shape[-3]
    clim, cmap = _get_colors(clim, cmap, data, component)
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
    _add_colorbar(cbar, component, fig, ax[0, :], imi, clim, title)
    _add_colorbar(cbar, diff_component, fig, ax[1, :], imd, difflim, 'Diff (%)')
    plt.close()
    return fig

def diff_matrix(fnames, dsets=['image'], titles=None, axis='z', slice_pos=0.5,
         other_dims=[], other_indices=[], img_offset=-1, img_slices=(slice(None), slice(None)),
         component='mag', clim=None, cmap=None, cbar=True,
         diff_component='real', difflim=None, diffmap=None, diffbar=True,
         rotates=0, fliplr=False, title=None):

    if len(dsets) == 1:
        dsets = dsets * len(fnames)
    if titles is not None and len(titles) != len(fnames):
        raise('Number of titles and files did not match')

    slice_dim, img_dims = _get_dims(axis, img_offset)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fn, 'r')) for fn in fnames]
        dsets = [f[dset] for f, dset in zip(files, dsets)]
        slice_index = int(np.floor(dsets[0].shape[slice_dim] * slice_pos))
        data = [_get_slices(D, slice_dim, slice(slice_index, slice_index+1), img_dims, img_slices, other_dims, other_indices) for D in dsets]
        data = np.concatenate(data)

    n = data.shape[-3]
    diffs = []
    for ii in range(1, n):
        diffs.append([])
        for jj in range(ii):
            diffs[ii - 1].append(100 * (data[ii, :, :] - data[jj, :, :]) / data[jj, :, :])

    clim, cmap = _get_colors(clim, cmap, data, component)
    difflim, diffmap = _get_colors(difflim, diffmap, diffs[0][0], diff_component)
    fig, ax = plt.subplots(n, n, figsize=(rc['figsize']*n, rc['figsize']*n), facecolor='black')
    for ii in range(n):
        imi = _draw(ax[ii, ii], _orient(np.squeeze(data[ii, :, :]), rotates, fliplr), component, clim, cmap)
        if titles is not None:
            ax[ii, ii].text(0.5, 0.9, titles[ii], color='white', transform=ax[ii, ii].transAxes, ha='center',
                            fontsize=rc['fontsize'], path_effects=rc['effects'])
        for jj in range(ii):
            imd = _draw(ax[jj, ii], _orient(np.squeeze(diffs[ii - 1][jj]), rotates, fliplr), diff_component, difflim, diffmap)
        for jj in range(ii, n):
            ax[jj, ii].set_facecolor('black')
            ax[jj, ii].axis('off')
    fig.subplots_adjust(wspace=0, hspace=0)

    if cbar:
        _add_colorbar(cbar, component, fig, imi, clim, title, ax=ax[0, 0])
    if diffbar:
        _add_colorbar(diffbar, diff_component, fig, imd, difflim, 'Diff (%)', ax=ax[0, -1])
    plt.close()
    return fig

def noncart(fname, dset='noncartesian', channels=slice(0), read_slice=slice(None), spoke_slice=slice(None),
            slab=0, volume=0,
            component='x', clim=None, cmap=None, cbar=True, title=None):
    with h5py.File(fname) as f:
        D = f[dset]
        data = _get_slices(D, -1, channels, (-1, -2), (read_slice, spoke_slice), (-3, -4), (slab, volume))

    n = data.shape[-3]
    clim, cmap = _get_colors(clim, cmap, data, component)
    fig, all_ax = plt.subplots(n, 1, figsize=(rc['figsize']*2, rc['figsize']*n))

    for ii in range(n):
        ax = _get_axes(all_ax, 0, ii)
        im = _draw(ax, np.squeeze(data[ii, :, :]).T, component, clim, cmap)
        _add_colorbar(cbar, component, fig, im, clim, title, ax)
    plt.close()
    return fig

def weights(filename, dset='sdc', sl_read=slice(None, None, 1), sl_spoke=slice(None, None, 1), log=False, clim=None):
    with h5py.File(filename) as f:
        data = np.array(f[dset][sl_spoke, sl_read]).T
        if log:
            data = np.log(data)
            if clim is None:
                clim = (np.log(1E-10), np.max(data))
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

def _get_slices(dset, slice_dim, slices, img_dims, img_slices, other_dims, other_indices):
    if dset.ndim < 4:
        raise Exception('Requires at least a 4D image')

    if len(other_indices) > (dset.ndim - 3):
        raise Exception('Too many other_indices')
    elif len(other_indices) < (dset.ndim - 3):
        other_indices.extend([0,]*(dset.ndim - 3 - len(other_indices)))
        other_dims.extend([x for x in range(-dset.ndim, 0) if x not in list([*img_dims, *other_dims, slice_dim])])
    
    all_slices = [slice(None),]*dset.ndim
    all_slices[img_dims[0]] = img_slices[0]
    all_slices[img_dims[1]] = img_slices[1]
    all_slices[slice_dim] = slices
    
    for (od, oi) in zip(other_dims, other_indices):
        all_slices[od] = slice(oi, oi+1)
    all_dims=(*other_dims, slice_dim, *img_dims)

    data = dset[tuple(all_slices)]
    data = data.transpose(all_dims)
    data = data.reshape(data.shape[-3], data.shape[-2], data.shape[-1])

    return data

def _get_dims(axis, offset):
    """
    Gives the image dims and slice dim for the given axis. Because of
    the Eigen/Numpy ordering swap, dimensions are counted backwards
    from the end.
    
    Args:
        axis - 'x', 'y' or 'z'
        offset - Dimension to count back from
    """
    if axis == 'z':
        slice_dim = offset - 3
        img_dims = [offset - 2, offset - 1]
    elif axis == 'y':
        slice_dim = offset - 2
        img_dims = [offset - 1, offset - 3]
    elif axis == 'x':
        slice_dim = offset - 1
        img_dims = [offset - 3, offset - 2]
    return slice_dim, img_dims

def _orient(img, rotates, fliplr):
    if rotates > 0:
        img = np.rot90(img, rotates)
    if fliplr:
        img = np.fliplr(img)
    return img

def _symmetrize_real(x):
    if x[0] < 0:
        x[1] = np.amax([np.abs(x[0]), np.abs(x[1])])
        x[0] = -x[1]
    return x

def _get_colors(clim, cmap, img, component):
    if not clim:
        if component == 'mag':
            clim = np.nanpercentile(np.abs(img), (2, 98))
        elif component == 'log':
            clim = np.nanpercentile(np.log1p(np.abs(img)), (2, 98))
        elif component == 'pha':
            clim = (-np.pi, np.pi)
        elif component == 'real':
            clim = _symmetrize_real(np.nanpercentile(np.real(img), (2, 98)))
        elif component == 'imag':
            clim = _symmetrize_real(np.nanpercentile(np.imag(img), (2, 98)))
        elif component == 'x':
            clim = np.nanpercentile(np.real(np.abs(img)), (2, 98))
        else:
            raise(f'Unknown component {component}')
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
        else:
            raise(f'Unknown component {component}')
    return (clim, cmap)

def _add_colorbar(cbar, component, fig, im, clim, title, ax=None, cax=None):
    if not cbar:
        return
    if component == 'x':
        _add_colorball(clim, ax=ax, cax=cax)
        if title is not None:
            fig.suptitle(title, color='white')
    else:
        if cax is None:
            cax = _first(ax).inset_axes(bounds=(0.1, 0.1, 0.8, 0.05), facecolor='black')
        cb = fig.colorbar(im, cax=cax, orientation='horizontal')
        axes = cb.ax
        ticks = (clim[0], np.sum(clim)/2, clim[1])
        tick_fmt='{:.4g}'
        labels = (tick_fmt.format(clim[0]), title, tick_fmt.format(clim[1]))
        cb.set_ticks(ticks)
        cb.set_ticklabels(labels, fontsize=rc['fontsize'], path_effects=rc['effects'])
        cb.ax.tick_params(axis='x', bottom=False, top=False)
        cb.ax.get_xticklabels()[0].set_ha('left')
        cb.ax.get_xticklabels()[1].set_ha('center')
        cb.ax.get_xticklabels()[2].set_ha('right')
        cb.ax.tick_params(color='w', labelcolor='w')

def _add_colorball(clim, ax=None, cax=None, tick_fmt='{:.1g}', cmap='cet_colorwheel'):
    if cax is None:
        cax = _first(ax).inset_axes(bounds=(0.01, 0.05, 0.4, 0.4), projection='polar', facecolor='black')
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
    cax.set_xticklabels([tick_fmt.format(clim[1]), tick_fmt.format(clim[1]) + 'i'],
                        fontsize=rc['fontsize'], path_effects=rc['effects'])
    cax.xaxis.set_tick_params(pad=10)
    cax.set_yticks([0, 1])
    cax.set_yticklabels([])

def _first(maybe_iterable):
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

def _draw(ax, data, component, clim, cmap):
    if component == 'x':
        _draw_x(ax, data, clim, cmap)
        return None
    elif component == 'mag':
        img = np.abs(data)
    elif component == 'log':
        img = np.log1p(np.abs(data))
    elif component == 'pha':
        img = np.angle(data)
    elif component == 'real':
        img = np.real(data)
    elif component == 'imag':
        img = np.imag(data)
    else:
        img = np.real(data)
        warnings.warn('Unknown component, taking real')
    im = ax.imshow(img, cmap=cmap, interpolation=rc['interpolation'], vmin=clim[0], vmax=clim[1])
    ax.axis('off')
    return im

def _draw_x(ax, img, clim, cmap='cet_colorwheel'):
    mag = np.real(np.abs(img))
    mag = np.clip((mag - clim[0]) / (clim[1] - clim[0]), 0, 1)

    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    pha = np.angle(img)
    colorized = smap.to_rgba(pha, alpha=1., bytes=False)[:, :, 0:3]
    colorized = colorized * mag[:, :, None]
    ax.imshow(colorized, interpolation=rc['interpolation'])
    ax.axis('off')

def basis(path, sl_spoke=slice(None), b=slice(None)):
    with h5py.File(path, 'r') as f:
        basis = f['basis'][b,sl_spoke]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(basis.T)
        ax.legend([str(x) for x in range(basis.shape[1])])
        plt.close()
        return fig

def dynamics(filename, sl=slice(None)):
    with h5py.File(filename) as f:
        dyn = f['dynamics'][sl,:]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(dyn.T)
        ax.grid('on')
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

def traj2d(filename, sl_read=slice(None), sl_spoke=slice(None), color='read', sps=None):
    with h5py.File(filename) as f:
        traj = np.array(f['trajectory'])
        fig, ax = plt.subplots(1, 1, figsize=(12, 4), facecolor='w')
        if color == 'read':
            c = np.tile(np.arange(len(traj[0, sl_read, 0])), len(traj[sl_spoke, 0, 0]))
        elif color == 'seg':
            c = np.tile(np.repeat(np.arange(sps),
                                  len(traj[0, sl_read, 0])),
                        int(len(traj[sl_spoke, 0, 0])/sps))
        else:
            c = np.tile(np.arange(len(traj[sl_spoke, 0, 0])), (len(traj[0, sl_read, 0]), 1)).ravel(order='F')
        ax.grid()
        ax.scatter(traj[sl_spoke, sl_read, 0],
                      traj[sl_spoke, sl_read, 1], c=c, s=0.5)
        ax.set_aspect('equal')
        fig.tight_layout()
        plt.close()
    return fig

def traj3d(filename, sl_read=slice(None), sl_spoke=slice(None), color='read', sps=None, angles=[30,-60,0]):
    with h5py.File(filename) as ff:
        traj = ff['trajectory'][sl_spoke, sl_read, :]
        if color == 'read':
            c = np.tile(np.arange(traj.shape[1], traj.shape[0]))
        elif color == 'seg':
            c = np.tile(np.repeat(np.arange(sps), traj.shape[1]), int(traj.shape[0]/sps))
        else:
            c = np.tile(np.arange(traj.shape[0]), (traj.shape[1], 1))
        fig = plt.figure(figsize=(12,12))
        ax = fig.add_subplot(projection='3d')

        x, y, z = np.array([[-0.5,0,0],[0,-0.5,0],[0,0,-0.5]])
        u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
        ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1)
        ax.scatter(traj[:, :, 0], traj[:, :, 1], traj[:, :, 2],
                   c=c, s=3, cmap='cmr.lavender')

        ax.view_init(elev=angles[0], azim=angles[1], vertical_axis='z')
        fig.tight_layout()
        plt.close()
    return fig
