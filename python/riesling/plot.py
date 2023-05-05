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
      'effects':([effects.Stroke(linewidth=4, foreground='black'), effects.Normal()])}

def planes(fname, dset='image',
           other_dims=None, other_indices=None, img_offset=-1,
           component='mag', clim=None, cmap=None, cbar=True,
           rotates=(0, 0, 0), fliplr=False, title=None,
           basis_file=None, basis_tp=0):

    dim_x, img_x = _get_dims('x', img_offset)
    dim_y, img_y = _get_dims('y', img_offset)
    dim_z, img_z = _get_dims('z', img_offset)
    with h5py.File(fname, 'r') as f:
        D = f[dset]
        index_x = int(np.floor(D.shape[dim_x] * 0.5))
        index_y = int(np.floor(D.shape[dim_y] * 0.5))
        index_z = int(np.floor(D.shape[dim_z] * 0.5))
        data_x = _get_slices(D, dim_x, [index_x], img_x, other_dims=other_dims, other_indices=other_indices, basis_file=basis_file, basis_tp=basis_tp)
        data_y = _get_slices(D, dim_y, [index_y], img_y, other_dims=other_dims, other_indices=other_indices, basis_file=basis_file, basis_tp=basis_tp)
        data_z = _get_slices(D, dim_z, [index_z], img_z, other_dims=other_dims, other_indices=other_indices, basis_file=basis_file, basis_tp=basis_tp)

    clim, cmap = _get_colors(clim, cmap, data_x, component)
    fig, ax = plt.subplots(1, 3, figsize=(rc['figsize']*3, rc['figsize']*1), facecolor='black')

    im_x = _draw(ax[0], _orient(np.squeeze(data_x), rotates[0], fliplr), component, clim, cmap)
    im_y = _draw(ax[1], _orient(np.squeeze(data_y), rotates[1], fliplr), component, clim, cmap)
    im_z = _draw(ax[2], _orient(np.squeeze(data_z), rotates[2], fliplr), component, clim, cmap)
    fig.tight_layout(pad=0)
    _add_colorbar(cbar, component, fig, im_x, clim, title, ax=ax[1])
    plt.close()
    return fig

def slices(fname, dset='image', n=4, axis='z', start=0.25, stop=0.75,
           other_dims=None, other_indices=None, img_offset=-1, img_slices=None,
           component='mag', clim=None, cmap=None, cbar=True,
           rows=1, rotates=0, fliplr=False, title=None,
           basis_file=None, basis_tp=0):

    slice_dim, img_dims = _get_dims(axis, img_offset)
    with h5py.File(fname, 'r') as f:
        D = f[dset]
        maxn = D.shape[slice_dim]
        n = np.amin([n, maxn])
        slices = np.floor(np.linspace(start*maxn, stop*maxn, n, endpoint=True)).astype(int)
        data = _get_slices(D, slice_dim, slices, img_dims, img_slices, other_dims, other_indices, basis_file=basis_file, basis_tp=basis_tp)

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

def series(fname, dset='image', axis='z', slice_pos=0.5, series_dim=-1, series_slice=None,
           other_dims=None, other_indices=None, img_offset=-1, img_slices=None, scales=None,
           component='mag', clim=None, cmap=None, cbar=True, rows=1, rotates=0, fliplr=False, title=None):
    slice_dim, img_dims = _get_dims(axis, img_offset)
    if series_slice is None:
        series_slice = slice(None)
    if other_dims is None:
        other_dims = [slice_dim]
    else:
        other_dims = [slice_dim, *other_dims]
    with h5py.File(fname, 'r') as f:
        D = f[dset]
        slice_index = int(np.floor(D.shape[slice_dim] * slice_pos))
        if other_indices is None:
            other_indices = [slice_index]
        else:
            other_indices = [slice_index, *other_indices]
        data = _get_slices(D, series_dim, series_slice, img_dims, img_slices, other_dims, other_indices)

    if scales is not None:
        data = data * np.array(scales).reshape([data.shape[0], 1, 1])

    n = data.shape[-3]
    clim, cmap = _get_colors(clim, cmap, data, component)
    cols = int(np.ceil(n / rows))
    fig, all_ax = plt.subplots(rows, cols, figsize=(rc['figsize']*cols, rc['figsize']*rows), facecolor='black')
    for ir in range(rows):
        for ic in range(cols):
            sl = (ir * cols) + ic
            ax = _get_axes(all_ax, ir, ic)
            if sl < n:
                im = _draw(ax, _orient(np.squeeze(data[sl, :, :]), rotates, fliplr), component, clim, cmap)
            else:
                ax.axis('off')

    fig.tight_layout(pad=0)
    _add_colorbar(cbar, component, fig, im, clim, title, ax=all_ax)
    plt.close()
    return fig

def sense(fname, dset='sense', **kwargs):
    return series(fname, dset=dset, component='x', **kwargs)

def diff(fnames, dsets=['image'], titles=None, axis='z', slice_pos=0.5,
         other_dims=None, other_indices=None, img_offset=-1, img_slices=None,
         component='mag', clim=None, cmap=None, cbar=False,
         diff_component='real', difflim=None, diffmap=None, diffbar=True,
         rotates=0, fliplr=False, title=None,
         basis_files=[None], basis_tps=[0]):

    if len(fnames) < 2:
        raise('Must have more than 1 image to diff')
    if len(dsets) == 1:
        dsets = dsets * len(fnames)
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

    slice_dim, img_dims = _get_dims(axis, img_offset)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fn, 'r')) for fn in fnames]
        dsets = [f[dset] for f, dset in zip(files, dsets)]
        slice_index = int(np.floor(dsets[0].shape[slice_dim] * slice_pos))
        data = [_get_slices(D, slice_dim, slice(slice_index, slice_index+1), img_dims, img_slices, other_dims, other_indices, basis_file=basis_file, basis_tp=basis_tp) for [D, basis_file, basis_tp] in zip(dsets, basis_files, basis_tps)]
        data = np.concatenate(data)
    ref = np.max(np.abs(data[:-1, :, :]))
    diffs = np.diff(data, n=1, axis=0) * 100 / ref
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
    _add_colorbar(cbar, component, fig, imi, clim, title, ax=ax[0, :])
    _add_colorbar(diffbar, diff_component, fig, imd, difflim, 'Diff (%)', ax=ax[1, :])
    plt.close()
    return fig

def diff_matrix(fnames, dsets=['image'], titles=None, axis='z', slice_pos=0.5,
         other_dims=None, other_indices=None, img_offset=-1, img_slices=None,
         component='mag', clim=None, cmap=None, cbar=True,
         diff_component='real', difflim=None, diffmap=None, diffbar=True,
         rotates=0, fliplr=False, title=None,
         basis_files=None, basis_tps=0):

    if len(fnames) < 2:
        raise('Must have more than 1 image to diff')
    if len(dsets) == 1:
        dsets = dsets * len(fnames)
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

    slice_dim, img_dims = _get_dims(axis, img_offset)
    with contextlib.ExitStack() as stack:
        files = [stack.enter_context(h5py.File(fn, 'r')) for fn in fnames]
        dsets = [f[dset] for f, dset in zip(files, dsets)]
        slice_index = int(np.floor(dsets[0].shape[slice_dim] * slice_pos))
        data = [_get_slices(D, slice_dim, slice(slice_index, slice_index+1), img_dims, img_slices, other_dims, other_indices, basis_file=basis_file, basis_tp=basis_tp) for [D, basis_file, basis_tp] in zip(dsets, basis_files, basis_tps)]
        data = np.concatenate(data)

    n = data.shape[-3]
    diffs = []
    ref = np.max(np.abs(data[0, :, :]))
    for ii in range(1, n):
        diffs.append([])
        for jj in range(ii):
            diffs[ii - 1].append((data[jj, :, :] - data[ii, :, :]) * 100 / ref)

    clim, cmap = _get_colors(clim, cmap, data, component)
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

def noncart(fname, dset='noncartesian', channels=slice(1), read_slice=slice(None), spoke_slice=slice(None),
            slab=0, volume=0, rows=1, component='xlog', clim=None, cmap=None, cbar=True, title=None, transpose=False):
    with h5py.File(fname) as f:
        D = f[dset]
        data = _get_slices(D, -1, channels, (-2, -3), (read_slice, spoke_slice), (-4, -5), (slab, volume))
    if transpose:
        data = np.transpose(data, axes=(0,2,1))
    n = data.shape[0]
    clim, cmap = _get_colors(clim, cmap, data, component)

    cols = int(np.ceil(n / rows))
    height = rc['figsize']*rows
    width = rc['figsize']*cols*data.shape[2]/data.shape[1]
    fig, all_ax = plt.subplots(rows, cols, figsize=(width, height), facecolor='black')
    for ir in range(rows):
        for ic in range(cols):
            sl = (ir * cols) + ic
            ax = _get_axes(all_ax, ir, ic)
            im = _draw(ax, np.squeeze(data[sl, :, :]), component, clim, cmap)
            _add_colorbar(cbar, component, fig, im, clim, title, ax)
    plt.close()
    return fig

def weights(filename, dset='sdc', sl_read=slice(None, None, 1), sl_spoke=slice(None, None, 1), log=False, clim=None):
    with h5py.File(filename) as f:
        data = np.array(f[dset][sl_spoke, sl_read]).T
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

def _get_slices(dset, slice_dim, slices, img_dims, img_slices=None, other_dims=None, other_indices=None,
                basis_file=None, basis_tp=0):
    if dset.ndim < 3:
        raise Exception('Requires at least a 3D image')
    if img_slices is None:
        img_slices = (slice(None), slice(None))
    if other_dims is None:
        other_dims = []
    if other_indices is None:
        other_indices = []
    if basis_file is None:
        basis_dim = 0
    else:
        with h5py.File(basis_file) as f:
            basis = f['basis'][basis_tp, :]
        basis_dim = -1

    if len(other_indices) > (dset.ndim - 3):
        raise Exception('Too many other_indices')
    elif len(other_indices) < (dset.ndim - 3):
        other_indices.extend([0,]*(dset.ndim - 3 - len(other_indices)))
        other_dims.extend([x for x in range(-dset.ndim, 0) if x not in list([*img_dims, *other_dims, slice_dim, basis_dim])])
    all_slices = [slice(None),]*dset.ndim
    all_slices[img_dims[0]] = img_slices[0]
    all_slices[img_dims[1]] = img_slices[1]
    all_slices[slice_dim] = slices
    
    for (od, oi) in zip(other_dims, other_indices):
        all_slices[od] = slice(oi, oi+1)
    all_dims=(*other_dims, slice_dim, *img_dims)

    data = dset[tuple(all_slices)]
    if basis_file:
        data = np.dot(data, basis)
        data = data.transpose([ii + 1 for ii in all_dims])
    else:
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
        slice_dim = offset - 1
        img_dims = [offset - 3, offset - 2]
    elif axis == 'x':
        slice_dim = offset - 2
        img_dims = [offset - 1, offset - 3]
    return slice_dim, img_dims

def _orient(img, rotates, fliplr=False):
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
            clim = np.nanpercentile(np.abs(img), (2, 99))
        elif component == 'log':
            clim = np.nanpercentile(np.log1p(np.abs(img)), (2, 98))
        elif component == 'pha':
            clim = (-np.pi, np.pi)
        elif component == 'real':
            clim = _symmetrize_real(np.nanpercentile(np.real(img), (2, 99)))
        elif component == 'imag':
            clim = _symmetrize_real(np.nanpercentile(np.imag(img), (2, 99)))
        elif component == 'x':
            clim = np.nanpercentile(np.abs(img), (2, 99))
        elif component == 'xlog':
            clim = np.nanpercentile(np.log1p(np.abs(img)), (2, 99))
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
            fig.text(0.5, 0.05, title, color='white', ha='center', fontsize=rc['fontsize'], path_effects=rc['effects'])
    elif component == 'x' or component == 'xlog':
        _add_colorball(clim, ax=ax, cax=cax)
        if title is not None:
            fig.text(0.5, 0.05, title, color='white', ha='center', fontsize=rc['fontsize'], path_effects=rc['effects'])
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
        labels = (f' {clim[0]:2.1g}', title, f'{clim[1]:2.1g} ')
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
    elif component == 'xlog':
        _draw_x(ax, data, clim, cmap, True)
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

def _draw_x(ax, img, clim, cmap='cet_colorwheel', log=False):
    mag = np.real(np.abs(img))
    if log:
        mag = np.log1p(mag);
    mag = np.clip((mag - clim[0]) / (clim[1] - clim[0]), 0, 1)

    norm = colors.Normalize(vmin=-np.pi, vmax=np.pi)
    smap = cm.ScalarMappable(norm=norm, cmap=cmap)
    pha = np.angle(img)
    colorized = smap.to_rgba(pha, alpha=1., bytes=False)[:, :, 0:3]
    colorized = colorized * mag[:, :, None]
    ax.imshow(colorized, interpolation=rc['interpolation'])
    ax.axis('off')

def basis(path, sl_spoke=slice(None), b=slice(None), show_sum=False):
    with h5py.File(path, 'r') as f:
        basis = f['basis'][sl_spoke,b]
        fig, ax = plt.subplots(figsize=(16, 6))
        ax.plot(basis)
        leg = [str(x) for x in range(basis.shape[1])]
        if show_sum:
            ax.plot(np.sum(basis, axis=1))
            leg.append('Sum')
        ax.legend(leg)
        ax.grid(True)
        ax.autoscale(enable=True, tight=True)
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

def traj2d(filename, read_slice=slice(None), spoke_slice=slice(None), color='read', sps=None):
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
            ax[ii].set_xlim((-0.5,0.5))
            ax[ii].set_ylim((-0.5,0.5))
        fig.tight_layout()
        plt.close()
    return fig

def traj3d(filename, read_slice=slice(None), spoke_slice=slice(None), color='read', sps=None, angles=[30,-60,0]):
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

        x, y, z = np.array([[-0.5,0,0],[0,-0.5,0],[0,0,-0.5]])
        u, v, w = np.array([[1,0,0],[0,1,0],[0,0,1]])
        ax.quiver(x,y,z,u,v,w,arrow_length_ratio=0.1)
        ax.scatter(traj[:, :, 0], traj[:, :, 1], traj[:, :, 2],
                   c=c, s=3, cmap='cmr.ember')
        ax.set_xlim((-0.5,0.5))
        ax.set_ylim((-0.5,0.5))
        ax.set_zlim((-0.5,0.5))
        ax.view_init(elev=angles[0], azim=angles[1], vertical_axis='z')
        fig.tight_layout()
        plt.close()
    return fig
